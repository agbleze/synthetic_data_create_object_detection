
#%%
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
import albumentations as A
import time
from tqdm import tqdm


obj_dict = {
    1: {"folder": "battery", "longest_min": 150, "logngest_max": 800},
    2: {"folder": "lightbulb", "longest_min": 150, "longest_max": 800},
    3: {"folder": "padlock", "longest_min": 150, "longest_max": 800}
}

PATH_MAIN = "data"

for k, _ in obj_dict.items():
    folder_name = obj_dict[k]["folder"]
    
    files_imgs = sorted(os.listdir(os.path.join(PATH_MAIN, folder_name, "images")))
    files_imgs = [os.path.join(PATH_MAIN, folder_name, "images", f) for f in files_imgs]
    
    files_masks = sorted(os.listdir(os.path.join(PATH_MAIN, folder_name, "masks")))
    files_masks = [os.path.join(PATH_MAIN, folder_name, "masks", f) for f in files_masks]
    
    obj_dict[k]["images"] = files_imgs
    obj_dict[k]['masks'] = files_masks
    
print("The first five files from the sorted list of battery images:", obj_dict[1]['images'][:5])
print("\nThe first five files from the sorted list of battery masks:", obj_dict[1]['masks'][:5])
    
files_bg_imgs = sorted(os.listdir(os.path.join(PATH_MAIN, "bg")))   
files_bg_imgs = [os.path.join(PATH_MAIN, "bg", f) for f in files_bg_imgs]

files_bg_noise_imgs = sorted(os.listdir(os.path.join(PATH_MAIN, "bg_noise", "images")))
files_bg_noise_imgs = [os.path.join(PATH_MAIN, "bg_noise", "images", f) for f in files_bg_noise_imgs]
files_bg_noise_masks = sorted(os.listdir(os.path.join(PATH_MAIN, "bg_noise", "masks")))
files_bg_noise_masks = [os.path.join(PATH_MAIN, "bg_noise", "masks", f) for f in files_bg_noise_masks]


#%%

def get_img_and_mask(img_path, mask_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    mask = cv2.imread(mask_path)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    
    mask_b = mask[:,:,0] == 0
    mask = mask_b.astype(np.uint8)
    
    
#%%
img_path = obj_dict[3]['images'][0]
mask_path = obj_dict[3]['masks'][0]

img, mask = get_img_and_mask(img_path, mask_path)

print("Image file:", img_path)
print("Mask file:", mask_path)
print("\nShape of the image of the object", img.shape)
print("Shape of the binary mask:", mask.shape)

fig, ax = plt.subplots(1, 2, figsize=(16, 7))
ax[0].imshow(img)
ax[0].set_title("Object", fontsize=18)

    
def resize_img(img, desired_max, desired_min=None):
    h, w = img.shape[0], img.shape[1]
    
    longest, shortest = max(h, w), min(h, w)
    longest_new = desired_max
    
    if desired_min:
        shortest_new = desired_min
    else:
        shortest_new = int(shortest * (longest_new / longest))
        
    if h > w:
        h_new, w_new = longest_new, shortest_new
    else:
        h_new, w_new = shortest_new, longest_new
        
    transform_resize = A.Compose([
        A.Sequential([
            A.Resize(h_new, w_new, interpolation=1, always_apply=False, p=1)
        ], p=1)
    ])
    
    transformed = transform_resize(image=img)
    img_r = transformed["image"]
    return img_r


#%%
img_bg_path = files_bg_imgs[5]
img_bg = cv2.imread(img_bg_path)
img_bg = cv2.cvtColor(img_bg, cv2.COLOR_BGRRGB)

img_bg_resized_1 = resize_img(img_bg, desired_max=1920, desired_min=None)
img_bg_resized_2 = resize_img(img_bg, desired_max=1920, desired_min=1080)

print("Shape of the original background image:", img_bg.shape)

print("Shape of the resized background image (desired_max=1920, desired_min=None):", img_bg_resized_1.shape)
print("Shape of the resized background image (desired_max=1920, desired_min=1080):", img_bg_resized_2.shape)

def plot(img_1, img_2):
    fig, ax = plt.subplots(1, 2, figsize=(16, 7))
    ax[0].imshow(img_1)
    ax[0].set_title("Resized (desired_max=1920, desired_min=None)", fontsize=18)
    ax[1].imshow(img_2)
    ax[1].set_title("Resized (desired_max=1920, desired_min=1080):", fontsize=18)
    plt.show()


plot(img_1=img_bg_resized_1, img_2=img_bg_resized_2)    

#%% resing and transforming objects
def resize_transform_obj(img, mask, longest_min, longest_max, transforms=False):
    h, w = mask.shape[0], mask.shape[1]
    
    longest, shortest = max(h, w), min(h, w)
    longest_new = np.random.randint(longest_min, longest_max)
    shortest_new = int(shortest * (longest_new/longest))
    
    if h > w:
        h_new, w_new = longest_new, shortest_new
    else:
        h_new, w_new = shortest_new, longest_new
        
    transform_resize = A.Resize(h_new, w_new, interpolation=1, always_apply=False, p=1)
    transformed_resized = transform_resize(image=img, mask=mask)
    img_t = transformed_resized["image"]
    mask_t = transformed_resized["mask"]
    
    if transforms:
        transformed = transforms(image=img_t, mask=mask_t)
        img_t = transformed["image"]
        mask_t = transformed["mask"]
        
    return img_t, mask_t


transforms_bg_obj = A.Compose([
    A.RandomRotate90(p=1),
    A.ColorJitter(brightness=0.3,
                  contrast=0.3,
                  saturation=0.3, 
                  hue=0.07,
                  always_apply=False,
                  p=1
                  ),
    A.Blur(blur_limit=(3,15),
           always_apply=False, p=0.5
           )
])

transform_obj = A.Compose([A.RandomRotate90(p=1),
                           A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.2),
                               contrast_limit=0.1, brightness_by_max=True,
                               always_apply=False, p=1
                               )
])
        
#%%
img_path = obj_dict[3]['images'][0]
mask_path = obj_dict[3]['masks'][0]

img, mask = get_img_and_mask(img_path, mask_path)        
img_t, mask_t = resize_transform_obj(img, mask, longest_min=300,
                                     longest_max=400,
                                     transforms=transform_obj
                                     )  
print("Shape of the image of the transformed object:", img_t.shape)
print("Shape of the transformed binary mask:", mask_t.shape)
print("\n")
  
    
         
        
    
    



