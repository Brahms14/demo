import torchvision
import pytorch_lightning as pl
import torchmetrics
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import torch
import os
class LitModel(pl.LightningModule):
    def __init__(self, num_classes, learning_rate=2e-4):
        super().__init__()
        
        # log hyperparameters
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.base=torchvision.models.resnet152(weights='DEFAULT')
        self.base.fc=nn.Linear(in_features=2048, out_features=num_classes, bias=True) 
      
        self.accuracy = torchmetrics.Accuracy(task="multiclass",num_classes= num_classes)
     

      
    # will be used during inference
    def forward(self, x):
        x=self.base(x)
        x = F.log_softmax(x, dim=1)
        return x
    
label_classes={45: 'Cavalier_king_charles_spaniel',
 92: 'Kerry_blue_terrier',
 71: 'German_shorthaired_pointer',
 105: 'Newfoundland',
 42: 'Canaan_dog',
 56: 'Dalmatian',
 108: 'Norwegian_elkhound',
 76: 'Gordon_setter',
 27: 'Bluetick_coonhound',
 39: 'Bulldog',
 69: 'German_pinscher',
 100: 'Maltese',
 10: 'Australian_cattle_dog',
 106: 'Norfolk_terrier',
 63: 'English_toy_spaniel',
 114: 'Papillon',
 35: 'Briard',
 80: 'Greyhound',
 9: 'Anatolian_shepherd_dog',
 117: 'Pembroke_welsh_corgi',
 113: 'Otterhound',
 6: 'American_foxhound',
 91: 'Keeshond',
 78: 'Great_pyrenees',
 109: 'Norwegian_lundehund',
 7: 'American_staffordshire_terrier',
 43: 'Cane_corso',
 122: 'Pomeranian',
 46: 'Chesapeake_bay_retriever',
 51: 'Clumber_spaniel',
 65: 'Field_spaniel',
 25: 'Black_russian_terrier',
 121: 'Pointer',
 118: 'Petit_basset_griffon_vendeen',
 84: 'Irish_red_and_white_setter',
 104: 'Neapolitan_mastiff',
 79: 'Greater_swiss_mountain_dog',
 52: 'Cocker_spaniel',
 103: 'Miniature_schnauzer',
 50: 'Chow_chow',
 62: 'English_springer_spaniel',
 33: 'Boxer',
 37: 'Brussels_griffon',
 34: 'Boykin_spaniel',
 64: 'Entlebucher_mountain_dog',
 16: 'Bearded_collie',
 89: 'Italian_greyhound',
 110: 'Norwich_terrier',
 30: 'Borzoi',
 58: 'Doberman_pinscher',
 102: 'Mastiff',
 128: 'Tibetan_mastiff',
 26: 'Bloodhound',
 24: 'Black_and_tan_coonhound',
 8: 'American_water_spaniel',
 66: 'Finnish_spitz',
 112: 'Old_english_sheepdog',
 75: 'Golden_retriever',
 32: 'Bouvier_des_flandres',
 49: 'Chinese_shar-pei',
 81: 'Havanese',
 18: 'Bedlington_terrier',
 41: 'Cairn_terrier',
 70: 'German_shepherd_dog',
 61: 'English_setter',
 68: 'French_bulldog',
 125: 'Saint_bernard',
 1: 'Afghan_hound',
 116: 'Pekingese',
 115: 'Parson_russell_terrier',
 12: 'Australian_terrier',
 126: 'Silky_terrier',
 20: 'Belgian_sheepdog',
 59: 'Dogue_de_bordeaux',
 131: 'Xoloitzcuintli',
 74: 'Glen_of_imaal_terrier',
 88: 'Irish_wolfhound',
 120: 'Plott',
 19: 'Belgian_malinois',
 87: 'Irish_water_spaniel',
 17: 'Beauceron',
 23: 'Bichon_frise',
 0: 'Affenpinscher',
 72: 'German_wirehaired_pointer',
 55: 'Dachshund',
 123: 'Poodle',
 83: 'Icelandic_sheepdog',
 14: 'Basset_hound',
 21: 'Belgian_tervuren',
 99: 'Lowchen',
 28: 'Border_collie',
 127: 'Smooth_fox_terrier',
 124: 'Portuguese_water_dog',
 36: 'Brittany',
 82: 'Ibizan_hound',
 29: 'Border_terrier',
 132: 'Yorkshire_terrier',
 101: 'Manchester_terrier',
 47: 'Chihuahua',
 31: 'Boston_terrier',
 53: 'Collie',
 107: 'Norwegian_buhund',
 22: 'Bernese_mountain_dog',
 97: 'Leonberger',
 94: 'Kuvasz',
 11: 'Australian_shepherd',
 2: 'Airedale_terrier',
 44: 'Cardigan_welsh_corgi',
 96: 'Lakeland_terrier',
 5: 'American_eskimo_dog',
 60: 'English_cocker_spaniel',
 3: 'Akita',
 90: 'Japanese_chin',
 38: 'Bull_terrier',
 85: 'Irish_setter',
 130: 'Wirehaired_pointing_griffon',
 86: 'Irish_terrier',
 4: 'Alaskan_malamute',
 111: 'Nova_scotia_duck_tolling_retriever',
 93: 'Komondor',
 13: 'Basenji',
 98: 'Lhasa_apso',
 54: 'Curly-coated_retriever',
 67: 'Flat-coated_retriever',
 15: 'Beagle',
 129: 'Welsh_springer_spaniel',
 48: 'Chinese_crested',
 77: 'Great_dane',
 119: 'Pharaoh_hound',
 73: 'Giant_schnauzer',
 95: 'Labrador_retriever',
 40: 'Bullmastiff',
 57: 'Dandie_dinmont_terrier'}
model=LitModel(num_classes=133).load_from_checkpoint(r"D:/app/app\model/best.ckpt")
transform = transforms.Compose([
              transforms.Resize((224, 224)), # resize to 224x224
                transforms.ToTensor(), # convert PIL image to Tensor
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]) # normalize to mean and std to Normal distribution
                ])

def predict(img:Image):
    img_transformed=transform(img)
    input_tensor=torch.unsqueeze(img_transformed,0)
    model.eval()
    logits=model(input_tensor)
    preds = torch.argmax(logits, dim=1)
    label_predicted=label_classes[int(preds.numpy())]
    
    return label_predicted
    
    


