from trt_pose.parse_objects import ParseObjects
from trt_pose.draw_objects import DrawObjects
from chainer import cuda, Variable, serializers
from PIL import Image, ImageFilter
from net import *
import torchvision.transforms as transforms
import cv2
import time
from torch2trt import TRTModule
import torch2trt
import torch
import trt_pose.models
import json
import trt_pose.coco
import os
import sys
import random

class Player:
    def __init__(self, heart_num, peaks, n_peaks, beginning, ending):
        self.heart_num = heart_num
        self.peaks = peaks
        self.n_peaks = n_peaks
        self.beginning = beginning
        self.ending = ending

def define_player(first_peaks, second_peaks):
    if torch.mean(first_peaks[:, 1]) <= torch.mean(second_peaks[:, 1]):
        return first_peaks, second_peaks # p1, p2
    else:
        return second_peaks, first_peaks

def warning_situation(player_num, all_peaks, p1_peaks, p2_peaks): # one, three people and cross line
    nonzero_peaks1 = torch.tensor([x for x in p1_peaks[:, 1] if x!=0])
    nonzero_peaks2 = torch.tensor([x for x in p2_peaks[:, 1] if x!=0])
    #if player_num == 1:
    #    if torch.sum(all_peaks[0, :, 0, :]) == 0 or torch.sum(all_peaks[0, :, 1, :]) != 0:
    #        return True
    if player_num == 2:
        if torch.sum(all_peaks[0, :, 1, :]) == 0:
            #print(1)
            return True
        #elif torch.sum(all_peaks[0, :, 2, :]) != 0:
        #    print(2)
        #    return True
        #elif torch.mean(nonzero_peaks1) > 0.45:
        #    print(torch.mean(nonzero_peaks1))
        #    print(3)
        #    cv2.waitKey(0)
        #    return True
        #elif torch.mean(nonzero_peaks2) < 0.55:
        #    print(torch.mean(nonzero_peaks2))
        #    print(4)
        #    cv2.waitKey(0)
        #    return True
    return False

def random_bomb(bomb_num):
    if bomb_num > max_bomb:
        bomb_num = max_bomb
    print(f'bomb number = {bomb_num}')
    bomb_list = []
    error = 0
    while len(bomb_list) < bomb_num:
        x1, y1 = random.randint(2 * half_explode_size, cap_width - 2 * half_explode_size), random.randint(2 * half_explode_size, cap_height - 2 * half_explode_size)
        for x, y in bomb_list:
            if math.sqrt((x - x1)**2 + (y - y1)**2) < 2 * half_explode_size:
                error += 1
                break
        if error == 0 or error > 5:
            bomb_list.append((x1, y1))
            error = 0
    return bomb_list

def detect_explosion(bomb,human):
    for keypoints in skeleton:
        point1 = human[keypoints[0]-1]
        point2 = human[keypoints[1]-1]
        if torch.all(point1)!=0 and torch.all(point2)!=0:
            for b in bomb:
                (tx1,ty1),(tx2,ty2),(x3,y3) = point1,point2,b
                x1 = ty1 * cap_width
                x2 = ty2 * cap_width
                y1 = tx1 * cap_height
                y2 = tx2 * cap_height
                if cal_distance(b, (x1, y1), (x2, y2)) <= circle_radius:
                    return True
    return False            

def cal_distance(bomb, point1, point2):
    bomb_arr = np.array(bomb)
    point1_arr = np.array(point1)
    point2_arr = np.array(point2)
    line_vector = point2_arr - point1_arr
    bomb_vector = bomb_arr - point1_arr

    line_vector_squared = np.dot(line_vector, line_vector)
    if line_vector_squared == 0:
        t = 0
    else:
        t = np.dot(bomb_vector, line_vector) / line_vector_squared

    t = max(0, min(1, t))
    closest_point = point1_arr + t * line_vector
    d = np.linalg.norm(bomb_arr - closest_point)
    return d

def game_start(hand_peak):
    #button_p1 = (cap_width*3//4, cap_height//4)
    #button_p2 = (cap_width//4, cap_height//4)
    button_p1 = (cap_width//4, cap_height*3//4)
    button_p2 = (cap_width//4, cap_height//4)
    x1, y1 = hand_peak
    x1 = x1*cap_width
    y1 = y1*cap_height
    for idx,b in enumerate([button_p1,button_p2]):
        x2, y2 = b
        length = math.sqrt((x1-x2)**2+(y1-y2)**2)
        if length <= 100:
            if op_sec[idx] < (duration * fps):
                op_sec[idx] += 1
        else:
            if op_sec[idx] > 0:
                op_sec[idx] -= 1
        
        if op_sec[idx] == (duration * fps):
            return idx+1

    return 0

if __name__ == '__main__':
    with open('human_pose.json', 'r') as f:
        human_pose = json.load(f)
    topology = trt_pose.coco.coco_category_to_topology(human_pose)
    num_parts = len(human_pose['keypoints'])
    num_links = len(human_pose['skeleton'])
    skeleton = human_pose['skeleton']
    
    OPTIMIZED_MODEL = 'resnet18_baseline_att_224x224_A_epoch_249_trt.pth'
    model_trt = TRTModule()
    model_trt.load_state_dict(torch.load(OPTIMIZED_MODEL))
    mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
    std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
    device = torch.device('cuda')

    def preprocess(image):
        global device
        device = torch.device('cuda')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = transforms.Resize((224, 224))(image)
        image = transforms.functional.to_tensor(image).to(device)
        image.sub_(mean[:, None, None]).div_(std[:, None, None])
        return image[None, ...]

    def normalize(objects):
        # objects : [18, 2]
        n_objects = torch.zeros(objects.shape)
        x_min = torch.min(objects[:, 0])
        x_range = torch.max(objects[:, 0]) - x_min
        y_min = torch.min(objects[:, 1])
        y_range = torch.max(objects[:, 1]) - y_min
        for i in range(objects.shape[0]):
            n_objects[i, 0] = (objects[i ,0] - x_min) / x_range
            n_objects[i, 1] = (objects[i ,1] - y_min) / y_range 
        return n_objects

    def detect_heart(objects):
        threshold = 0.1
        with open(f'bomb/heart.json', 'r') as f:
            r_data = json.load(f)
        reference = torch.tensor(r_data['keypoints'])
        r_mask = torch.tensor(r_data['mask'])
        loss = (torch.square(objects - reference).sum(dim=1) * r_mask).sum() / (torch.sum(r_mask).item())
        #print(loss)
        if loss <= threshold:
            return True
        else:
            return False

    parse_objects = ParseObjects(topology)
    draw_objects = DrawObjects(topology)
    #cap = cv2.VideoCapture('rtmp://140.116.56.6:1935/live')
    cap = cv2.VideoCapture(0)
    cap_height = 720
    cap_width = 1280
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)
    sec = 0
    op_sec = [0, 0]
    if_hurt = [False, False]
    fps = 10
    duration = 2
    angle_step = 360 / (duration * fps)
    bomb_duration = fps
    bomb_size = 84
    half_bomb_size = int(bomb_size/2)
    explode_size = 120
    half_explode_size = int(explode_size/2)
    circle_radius = 50
    max_bomb = 5
    player_life = 3
    bomb_path = './bomb_images/bomb.png'
    bomb_img = cv2.imread(bomb_path)
    bomb_img = cv2.resize(bomb_img, (bomb_size, bomb_size))
    explode_path = './bomb_images/explode.png'
    explode_img = cv2.imread(explode_path)
    explode_img = cv2.resize(explode_img, (explode_size, explode_size))
    already_start = False
    #player_num = 1

    round = 1
    print(f'Round {round}')
    bomb_dist_list = random_bomb(round)

    zero_peaks = np.zeros((1, 18, 1, 2))
    zero_npeaks = np.zeros((1, 18, 1, 2))
    init_beginning = time.time()
    init_ending = time.time()
    player_list = []
    p1 = Player(player_life, zero_peaks, zero_npeaks, init_beginning, init_ending)
    player_list.append(p1)
    #if player_num == 2:
    p2 = Player(player_life, zero_peaks[0, :, 0, :], zero_npeaks[0, :, 0, :], init_beginning, init_ending)
    player_list.append(p2)
    while(True):
        ret, image = cap.read()
        if ret:
            image = cv2.resize(image, (cap_width, cap_height))
            data = preprocess(image)
            cmap, paf = model_trt(data)
            cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
            counts, objects, peaks = parse_objects(cmap, paf)
            peaks_1, peaks_2 = define_player(peaks[0, :, 0, :], peaks[0, :, 1, :])
            draw_objects(image, counts, objects, peaks)
            p1.peaks = peaks[0, :, 0, :]
            if not already_start:
                player_num = game_start(p1.peaks[10]) 
            if player_num==0: 
                current_angle_1 = 360 - int(360 - op_sec[1] * angle_step)
                current_angle_2 = 360 - int(360 - op_sec[0] * angle_step)
                image = cv2.flip(image, 1)
                cv2.circle(image, (cap_width*3//4,cap_height//4), 100, (0,0,0), 5)
                cv2.ellipse(image, (cap_width*3//4,cap_height//4), (100, 100), -90, 0, current_angle_1, (5, 209, 255), thickness=5)

                text1 = f'1 player'
                (text_width, text_height), baseline = cv2.getTextSize(text1, cv2.FONT_HERSHEY_DUPLEX, 1, 10)
                (text_width2, text_height2), baseline = cv2.getTextSize(text1, cv2.FONT_HERSHEY_DUPLEX, 1, 3)
                image_h, image_w, _ = image.shape
                image_x = (image_w//2 - text_width) // 2
                image_y1 = image_h // 4
                image_y2 = image_h // 4
                cv2.putText(image, text1, (image_x, image_y1), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 10)
                cv2.putText(image, text1, (image_x, image_y2), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 3)
                
                cv2.circle(image, (cap_width//4,cap_height//4), 100, (0,0,0), 5)
                cv2.ellipse(image, (cap_width//4,cap_height//4), (100, 100), -90, 0, current_angle_2, (5, 209, 255), thickness=5)
                text2 = f'2 players'
                (text_width, text_height), baseline = cv2.getTextSize(text2, cv2.FONT_HERSHEY_DUPLEX, 1, 10)
                (text_width2, text_height2), baseline = cv2.getTextSize(text2, cv2.FONT_HERSHEY_DUPLEX, 1, 3)
                image_h, image_w, _ = image.shape
                image_x = (image_w//2 - text_width) // 2 + image_w//2
                image_y1 = image_h // 4
                image_y2 = image_h // 4
                cv2.putText(image, text2, (image_x, image_y1), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 10)
                cv2.putText(image, text2, (image_x, image_y2), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 3)

                image = cv2.flip(image, 1)


            if player_num!=0:
                already_start = True
                p1.n_peaks = normalize(p1.peaks)
                player_heart = []
                player_heart.append(detect_heart(p1.n_peaks))
                if player_num == 2:
                    p1.peaks, p2.peaks = define_player(peaks[0, :, 0, :], peaks[0, :, 1, :])
                    p1.n_peaks = normalize(p1.peaks)
                    p2.n_peaks = normalize(p2.peaks)
                    player_heart = []
                    player_heart.append(detect_heart(p1.n_peaks))
                    player_heart.append(detect_heart(p2.n_peaks))
                if_warning = warning_situation(player_num, peaks, p1.peaks, p2.peaks)
                if not if_warning:
                    for idx, d_heart in enumerate(player_heart):
                        if d_heart:
                            player_list[idx].ending = time.time()
                            if player_list[idx].ending - player_list[idx].beginning >= 3:
                                player_list[idx].heart_num += 1
                                print(f'player{idx+1} get heart')
                                if player_list[idx].heart_num > 3:
                                    player_list[idx].heart_num = 3
                                player_list[idx].beginning = time.time()
                        else:
                            player_list[idx].beginning = time.time()

                    sec += 1
                    current_angle = int(360 - sec * angle_step)
                    if player_num ==1:
                        for bomb_dist in bomb_dist_list:
                            if current_angle >= 10:
                                #add bomb
                                roi = image[bomb_dist[1]-half_bomb_size:bomb_dist[1]+half_bomb_size, bomb_dist[0]-half_bomb_size:bomb_dist[0]+half_bomb_size]
                                img2gray = cv2.cvtColor(bomb_img, cv2.COLOR_BGR2GRAY)
                                _, mask = cv2.threshold(img2gray, 1, 255, cv2.THRESH_BINARY)
                                mask_inv = cv2.bitwise_not(mask)
                                img1_bg = cv2.bitwise_and(roi,roi,mask = mask)
                                img2_fg = cv2.bitwise_and(bomb_img,bomb_img,mask = mask_inv)
                                dst = cv2.add(img1_bg,img2_fg)
                                image[bomb_dist[1]-half_bomb_size:bomb_dist[1]+half_bomb_size, bomb_dist[0]-half_bomb_size:bomb_dist[0]+half_bomb_size] = dst
                                # add circle
                                cv2.ellipse(image, bomb_dist, (circle_radius, circle_radius), -30, 0, current_angle, (0, 0, 255), thickness=5)
                                cv2.ellipse(image, bomb_dist, (circle_radius, circle_radius), -30, 0, current_angle - 10, (0, 0, 0), thickness=5)
                            else:
                                # add explode
                                roi = image[bomb_dist[1]-half_explode_size:bomb_dist[1]+half_explode_size, bomb_dist[0]-half_explode_size:bomb_dist[0]+half_explode_size]
                                img2gray = cv2.cvtColor(explode_img, cv2.COLOR_BGR2GRAY)
                                _, mask = cv2.threshold(img2gray, 1, 255, cv2.THRESH_BINARY)
                                mask_inv = cv2.bitwise_not(mask)
                                img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
                                img2_fg = cv2.bitwise_and(explode_img,explode_img,mask = mask)
                                dst = cv2.add(img1_bg,img2_fg)
                                image[bomb_dist[1]-half_explode_size:bomb_dist[1]+half_explode_size, bomb_dist[0]-half_explode_size:bomb_dist[0]+half_explode_size] = dst

                        # explosion
                        if sec > (duration * fps) and sec <= ((duration * fps) + bomb_duration) and if_hurt[0] == False:
                            if_explode = detect_explosion(bomb_dist_list,p1.peaks)
                            if if_explode:
                                p1.heart_num -= 1
                                if_hurt[0] = True
                            #print(if_explode)

                        if p1.heart_num <= 0:
                            image = cv2.flip(image, 1) 
                            text = f'Finished in round {round}!'
                            (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 3, 5)
                            image_h, image_w, _ = image.shape
                            image_x = (image_w - text_width) // 2
                            image_y = (image_h + text_height) // 2
                            cv2.putText(image, text, (image_x, image_y), cv2.FONT_HERSHEY_DUPLEX, 3, (0, 0, 255), 10)
                            #image = cv2.flip(image, 1)
                            cv2.imshow('bomb battle', image)
                            cv2.waitKey(0)
                            break

                        #print(p1.heart_num)
                        image = cv2.flip(image, 1)

                        text = f'Player 1 life : {player_list[0].heart_num}'
                        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 1, 10)
                        (text_width2, text_height2), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 1, 3)
                        image_h, image_w, _ = image.shape
                        image_x = (image_w - text_width) // 2
                        image_y1 = text_height
                        image_y2 = text_height2
                        cv2.putText(image, text, (image_x, image_y1), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 10)
                        cv2.putText(image, text, (image_x, image_y2), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 3)
                        image = cv2.flip(image, 1)
                        # after explosion
                        if sec > ((duration * fps) + bomb_duration):
                            sec = 0
                            if_hurt[0] = False
                            round += 1
                            print(f'Round {round}')
                            bomb_dist_list = random_bomb(round)
                    elif player_num == 2:
                        # two players
                        image = cv2.flip(image, 1)
                        cv2.line(image, (cap_width // 2, 0), (cap_width // 2, cap_height), (0, 0, 255), 10)
                        image = cv2.flip(image, 1)
                        p2_bomb_dist_list = [[], []]
                        for b in bomb_dist_list:
                            p2_bomb_dist_list[0].append((b[0] // 2, b[1]))
                        for b in bomb_dist_list:
                            p2_bomb_dist_list[1].append((b[0] // 2 + cap_width // 2, b[1]))

                        for p in range(player_num):
                            # bomb
                            for bomb_dist in p2_bomb_dist_list[p]:
                                if current_angle >= 10:
                                    #add bomb
                                    roi = image[bomb_dist[1]-half_bomb_size:bomb_dist[1]+half_bomb_size, bomb_dist[0]-half_bomb_size:bomb_dist[0]+half_bomb_size]
                                    img2gray = cv2.cvtColor(bomb_img, cv2.COLOR_BGR2GRAY)
                                    _, mask = cv2.threshold(img2gray, 1, 255, cv2.THRESH_BINARY)
                                    mask_inv = cv2.bitwise_not(mask)
                                    img1_bg = cv2.bitwise_and(roi,roi,mask = mask)
                                    img2_fg = cv2.bitwise_and(bomb_img,bomb_img,mask = mask_inv)
                                    dst = cv2.add(img1_bg,img2_fg)
                                    image[bomb_dist[1]-half_bomb_size:bomb_dist[1]+half_bomb_size, bomb_dist[0]-half_bomb_size:bomb_dist[0]+half_bomb_size] = dst
                                    # add circle
                                    cv2.ellipse(image, bomb_dist, (circle_radius, circle_radius), -30, 0, current_angle, (0, 0, 255), thickness=5)
                                    cv2.ellipse(image, bomb_dist, (circle_radius, circle_radius), -30, 0, current_angle - 10, (0, 0, 0), thickness=5)
                                else:
                                    # add explode
                                    roi = image[bomb_dist[1]-half_explode_size:bomb_dist[1]+half_explode_size, bomb_dist[0]-half_explode_size:bomb_dist[0]+half_explode_size]
                                    img2gray = cv2.cvtColor(explode_img, cv2.COLOR_BGR2GRAY)
                                    _, mask = cv2.threshold(img2gray, 1, 255, cv2.THRESH_BINARY)
                                    mask_inv = cv2.bitwise_not(mask)
                                    img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
                                    img2_fg = cv2.bitwise_and(explode_img,explode_img,mask = mask)
                                    dst = cv2.add(img1_bg,img2_fg)
                                    image[bomb_dist[1]-half_explode_size:bomb_dist[1]+half_explode_size, bomb_dist[0]-half_explode_size:bomb_dist[0]+half_explode_size] = dst
                            
                            # explosion
                            if sec > (duration * fps) and sec <= ((duration * fps) + bomb_duration) and if_hurt[p] == False:
                                if_explode = detect_explosion(p2_bomb_dist_list[p], player_list[p].peaks)
                                if if_explode:
                                    player_list[p].heart_num -= 1
                                    if_hurt[p] = True
                        

                        heart_num_list = [player_list[0].heart_num, player_list[1].heart_num]
                        live_idx = [idx for idx, value in enumerate(heart_num_list) if value!=0]
                        if len(live_idx) == 0:
                            image = cv2.flip(image, 1) 
                            text = 'Draw! One more round!'
                            (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 3, 5)
                            image_h, image_w, _ = image.shape
                            image_x = (image_w - text_width) // 2
                            image_y = (image_h + text_height) // 2
                            cv2.putText(image, text, (image_x, image_y), cv2.FONT_HERSHEY_DUPLEX, 3, (0, 0, 255), 10)
                            #image = cv2.flip(image, 1)
                            cv2.imshow('bomb battle', image)
                            cv2.waitKey(0)
                            break
                            
                        elif len(live_idx) == 1:
                            image = cv2.flip(image, 1) 
                            text = f'The winner is player{live_idx[0] + 1}!'
                            (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 3, 5)
                            image_h, image_w, _ = image.shape
                            image_x = (image_w - text_width) // 2
                            image_y = (image_h + text_height) // 2
                            cv2.putText(image, text, (image_x, image_y), cv2.FONT_HERSHEY_DUPLEX, 3, (0, 0, 255), 10)
                            #image = cv2.flip(image, 1)
                            cv2.imshow('bomb battle', image)
                            cv2.waitKey(0)
                            break

                        #print(f'p1: {player_list[0].heart_num}')
                        #print(f'p2: {player_list[1].heart_num}')
                       
                        image = cv2.flip(image, 1)
                        cv2.putText(image, f'Player 1 life : {player_list[1].heart_num}', (cap_width // 12, cap_height // 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 10)
                        cv2.putText(image, f'Player 2 life : {player_list[0].heart_num}', (cap_width // 2 + cap_width // 12, cap_height // 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 10)
                        cv2.putText(image, f'Player 1 life : {player_list[1].heart_num}', (cap_width // 12, cap_height // 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 3)
                        cv2.putText(image, f'Player 2 life : {player_list[0].heart_num}', (cap_width // 2 + cap_width // 12, cap_height // 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 3)

                        # after explosion
                        if sec > ((duration * fps) + bomb_duration):
                            sec = 0
                            if_hurt = [False, False]
                            round += 1
                            print(f'Round {round}')
                            bomb_dist_list = random_bomb(round)
                        
                        image = cv2.flip(image, 1)
                    image = cv2.flip(image, 1)
                    text = f'Round {round}'
                    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 2, 10)
                    (text_width2, text_height2), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 2, 5)
                    image_h, image_w, _ = image.shape
                    image_x = (image_w - text_width) // 2
                    image_y1 = image_h - text_height
                    image_y2 = image_h - text_height2
                    cv2.putText(image, text, (image_x, image_y1), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 0), 10)
                    cv2.putText(image, text, (image_x, image_y2), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 255), 5)
                    image = cv2.flip(image, 1)

                else:
                    image = cv2.flip(image, 1) 
                    text = 'WARNING'
                    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 8,5)
                    image_h, image_w, _ = image.shape
                    image_x = (image_w - text_width) // 2
                    image_y = (image_h + text_height) // 2
                    cv2.putText(image, text, (image_x, image_y), cv2.FONT_HERSHEY_DUPLEX, 8, (0, 0, 255), 10)
                    #cv2.putText(image, f'WARNING', (cap_width // 2, cap_height // 2), cv2.FONT_HERSHEY_DUPLEX, 8, (0, 0, 255), 5)
                    image = cv2.flip(image, 1)
                    
            image = cv2.flip(image, 1)
            cv2.imshow('bomb battle', image)
            cv2.waitKey(1)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
