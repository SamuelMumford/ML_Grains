import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image
import matplotlib.pyplot as plt


class SEM_PFD_Final(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.ImagePath = "PFD_Fin_Img"
        self.ImagePath2 = "PFD_Fin_Img2"
        self.ImagePath3 = "PFD_Fin_Img3"
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PFD_Fin_Img"))))

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, self.ImagePath, "Image" + str(idx)+ '.csv')
        img_path2 = os.path.join(self.root, self.ImagePath2, "Image" + str(idx)+ '.csv')
        img_path3 = os.path.join(self.root, self.ImagePath3, "Image" + str(idx)+ '.csv')
        img0 = np.loadtxt(img_path, dtype = int, delimiter=',')
        img0 = img0/255
        img1 = np.loadtxt(img_path2, dtype = int, delimiter=',')
        img1 = img1/255
        img2 = np.loadtxt(img_path3, dtype = int, delimiter=',')
        img2 = img2/255
        img = np.stack((img0, img1, img2), axis = 2)
        masks = np.zeros(img0.shape)
        boxes =  []
        
        num_objs = 0

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = []
        area = torch.as_tensor(area, dtype=torch.float32)
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes.int()
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area.int()
        target["iscrowd"] = iscrowd
        #img = torch.from_numpy(img).double()
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        img = img.float()
        return img, target

    def __len__(self):
        return len(self.imgs)

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

      
def get_instance_segmentation_model(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)
    return model
    
import utils
from engine import train_one_epoch, evaluate
import transforms as T
    
def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    return T.Compose(transforms)


if __name__ == "__main__":
    base = 'C:/Users/sammy/Downloads/SEM/SEM/'
    dataset = SEM_PFD_Final(base, get_transform(train=False))

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # our dataset has two classes only - background and person
    num_classes = 2
    parts = 7
    
    # get the model using our helper function
    PATHFull = 'C:/Users/sammy/Downloads/SEM/SEM/MaskRCNNTrySherFullModel_3part2.pt'
    model = torch.load(PATHFull, map_location=torch.device('cpu'))
    model.eval()
    img, info = dataset[0]
    with torch.no_grad():
        prediction = model([img.to(device)])
    l = img.shape[1]
    w = img.shape[2]
    predictMask = np.zeros((l*parts, w*parts))
    trueMask = np.zeros((l*parts, w*parts))
    cutVal = .5
    for i in range(0, len(dataset)):
        print('part ' + str(i+1) + ' of ' + str(parts**2))
        a = i//parts
        b = i%parts
        img, _ = dataset[i]
        with torch.no_grad():
            prediction = model([img.to(device)])
        masks = prediction[0]['masks']
        BoolMask = 1.0*(masks > .5)
        Predict = torch.sum(BoolMask, axis = 0)
        Predict = 1.0*(Predict > .5)
        with torch.no_grad():
            pred = Predict[0].numpy()
        predictMask[a*l:(a+1)*l, b*w:(b+1)*w] = pred
plt.plot(predictMask)
plt.show()
fname = "C:/Users/sammy/Downloads/SEM/SEM/PredictMaskSher3dpt2_imgQiSmall1.csv"
np.savetxt(fname, predictMask, fmt = '%d', delimiter=',')