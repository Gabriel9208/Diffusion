import cv2

class LetterBox:
    def __init__(self, target_size=(64, 64), fill_color=(127, 127, 127)):
        self.target_size = target_size
        self.fill_color = fill_color
    
    def __call__(self, img):
        img_h, img_w = img.shape[:2]
        target_h, target_w = self.target_size

        scale = min(target_w / img_w, target_h / img_h)
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)
        
        resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        pad_w = target_w - new_w
        pad_h = target_h - new_h
        
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        
        padded_img = cv2.copyMakeBorder(
            resized_img, top, bottom, left, right, 
            cv2.BORDER_CONSTANT, value=self.fill_color
        )
        return padded_img