from ultralytics import YOLO
from pathlib import Path
import torch
from PIL import Image
import numpy
from src.config import load_config

import traceback

class YOLOv11Inference():
    def __init__(self,model_name, device='cuda'):
        
        self.model = YOLO(model_name)
        self.device=device
        self.model.to(self.device)
        # model:
        #   yolo : "example_Yolov11m.pt"
        #   conf_threshold : 0.3

        # data:
        #   img_ext : [".png" , ".jpeg" , ".HEIC" , ".jpg"]
        
        
        #loading config from default.yaml
        
        config = load_config()
        self.conf = config["model"]["conf_threshold"]
        self.valid_ext= config["data"]["img_ext"]
        
  
    
    def analyze_image(self, image_path):
        
        #Inferencing
        
        results = self.model.predict(   
            source=image_path,
            conf = self.conf,
            device=self.device
        )
        
        #Extracting from results
        
        detect = [] #info for image
        numclasses={}  #looks like 0:2, 1:3, 2:4 where 0 means person, 1 means smthig etc
        
        for each in results:
            for box in each.boxes:
                
                cls = each.names[int(box.cls)]
                
                confidence = float(box.conf)
                
                coord = box.xyxy[0].tolist()
                
                detect.append({
                    'class':cls,
                    'confidence':confidence,
                    'coord' : coord,
                    'count' : 1
                    
                })
                
                numclasses[cls]=numclasses.get(cls,0)+1 #if we get a class count for some class then we want to save its presence
                
                
        for det in detect:
            det['count'] = numclasses[det['class']]
            
        return {
            'image_path' : str(image_path),
            'detections' : detect,
            'total_obj' : len(detect),
            'unique_classes': list(numclasses.keys()), #which classes are present
            'class_counts' : numclasses # freq of each class
        }
    
    def analyze_dir(self,dir_path):
        DATA=[]
        patterns = [f"*{ext}" for ext in self.valid_ext]
        
        img_paths=[]
        for pattern in patterns:
            img_paths.extend(Path(dir_path).glob(pattern)) #to add files with the valid extensions in the list
            
            
        for path in img_paths:
            try:
                DATA.append(self.analyze_image(path)) #that method will return smhting that will be appended here
            except Exception as e:
                traceback.print_exc()
                print(f"Error : {str(e)}")
                    
                
        return DATA
                