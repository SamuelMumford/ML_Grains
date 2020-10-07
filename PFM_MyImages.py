import os
import numpy as np
import torch
import torch.utils.data


class SEM_PFD(torch.utils.data.Dataset):
    def __init__(self, root, TrVal, transforms=None):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        if(TrVal):
            self.ImagePath = "PFD_Test_Img"
            self.ImagePath2 = "PFD_Test_Img2"
            self.ImagePath3 = "PFD_Test_Img3"
            self.MasksPath = "PFD_Test_Masks"
            self.boxPath = "PFD_Test_boxes"
            self.imgs = list(sorted(os.listdir(os.path.join(root, "PFD_Test_Img"))))
            self.msks = list(sorted(os.listdir(os.path.join(root, "PFD_Test_Masks"))))
            self.bxs = list(sorted(os.listdir(os.path.join(root, "PFD_Test_boxes"))))
        else:
            self.ImagePath = "PFD_Train_Img"
            self.ImagePath2 = "PFD_Train_Img2"
            self.ImagePath3 = "PFD_Train_Img3"
            self.MasksPath = "PFD_Train_Masks"
            self.boxPath = "PFD_Train_boxes"
            self.imgs = list(sorted(os.listdir(os.path.join(root, "PFD_Train_Img"))))
            self.msks = list(sorted(os.listdir(os.path.join(root, "PFD_Train_Masks"))))
            self.bxs = list(sorted(os.listdir(os.path.join(root, "PFD_Train_boxes"))))

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, self.ImagePath, self.imgs[idx])
        img_path2 = os.path.join(self.root, self.ImagePath2, self.imgs[idx])
        img_path3 = os.path.join(self.root, self.ImagePath3, self.imgs[idx])
        mask_path = os.path.join(self.root, self.MasksPath, self.msks[idx])
        box_path = os.path.join(self.root, self.boxPath, self.bxs[idx])
        img0 = np.loadtxt(img_path, dtype = int, delimiter=',')
        img0 = img0/255
        img1 = np.loadtxt(img_path2, dtype = int, delimiter=',')
        img1 = img1/255
        img2 = np.loadtxt(img_path3, dtype = int, delimiter=',')
        img2 = img2/255
        img = np.stack((img0, img1, img2), axis = 2)
        masks = np.load(mask_path, allow_pickle=True)
        boxes =  np.loadtxt(box_path, dtype=int, delimiter=',')
        
        num_objs = boxes.shape[0]

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
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
    dataset = SEM_PFD(base, True, get_transform(train=False))
    dataset_test = SEM_PFD(base, False, get_transform(train=False))
    
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=0,
        collate_fn=utils.collate_fn)
    
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=utils.collate_fn)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # our dataset has two classes only - background and person
    num_classes = 2
    
    # get the model using our helper function
    model = get_instance_segmentation_model(num_classes)
    # move model to the right device
    model.to(device) 
    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.8, weight_decay=0.0003)
    # and a learning rate scheduler, usually step of 4
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=4,
                                                   gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 14

    for epoch in range(num_epochs):
        #train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        #update the learning rate
        lr_scheduler.step()
        #evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)
    
    PATH = 'C:/Users/sammy/Downloads/SEM/SEM/MaskRCNNTry_3part.pt'
    PATHFull = 'C:/Users/sammy/Downloads/SEM/SEM/MaskRCNNTry_3partFullModel.pt'
    torch.save(model.state_dict(), PATH)
    torch.save(model, PATHFull)