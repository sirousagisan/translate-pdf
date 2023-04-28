import os
import re
import glob
import copy
import tqdm
import shutil
import string
import random
import torch
import numpy as np
import pypdf
from pdf2image import convert_from_path
from PIL import Image, ImageDraw, ImageFont
from paddleocr import PPStructure, PaddleOCR
from transformers import pipeline
from util import fw_fill


class PdfTranslate:
    def __init__(self):
        self.dpi = 300
        self.font_size = 32
        self.model_name = "staka/fugumt-en-ja"
        self.currnt_dir = os.getcwd()
        self.model_path = os.path.join(self.currnt_dir, "model", self.model_name)
        self.re_sub = re.compile(r"- ")
        self.font_path = os.path.join(self.currnt_dir,"model" ,"SourceHanSerifK-Light.otf")
        self.font = ImageFont.truetype(self.font_path, size=self.font_size)
        self.folder = None
        
        # dl model 
        if torch.cuda.is_available():
            self.device = "cuda:0"
        else:
            self.device = "cpu"

        self.pipe = pipeline("translation", model=self.model_path, device=self.device)
        self.layout_model = PPStructure(table=False, ocr=False, lang="en")
        self.ocr_model = PaddleOCR(ocr=True, lang="en", ocr_version="PP-OCRv3")
    
    def _translate(self, text):
        text = self.pipe(text)[0]["translation_text"]
        return text
        # end
    
    def _translate_a_page(self, image, num=0):
        target_img = copy.deepcopy(image)
        layout_img = self.layout_model(np.array(target_img, dtype=np.uint8))
        target_img = np.array(target_img)
        for part in layout_img:
            if part["type"] == "text":
                ocr_text = list(map(lambda x: x, self.ocr_model(part["img"])))[1]
                tmp_text = " ".join(list(s[0] for s in ocr_text))
                cleaned_text = self.re_sub.sub("", tmp_text)
                translated_text = self._translate(cleaned_text)
                # make translate img blok 
                block = Image.new("RGB",
                            (
                                part["bbox"][2] - part["bbox"][0] + 7,
                                part["bbox"][3] - part["bbox"][1] ,
                            ),
                            color=(255, 255, 255)
                        )
                canvas = ImageDraw.Draw(block)
                p_text = fw_fill(translated_text, width=int(
                                (part["bbox"][2] - part["bbox"][0]) / (32 / 2) -1
                            ))
                canvas.text(
                        (0, 0),
                        text=p_text,
                        font=self.font,
                        fill=(0, 0, 0)
                    )
                target_img[
                        int(part["bbox"][1]) : int(part["bbox"][3]),
                        int(part["bbox"][0]) : int(part["bbox"][2] + 7),
                    ] = np.array(block)
        img = Image.fromarray(target_img)
        img.save(f"{self.folder}/{num}.pdf")

    def translate_pdf(self, path):
        n = random.randint(5, 10)
        self.folder = ''.join(random.choice(string.ascii_uppercase) for _ in range(n))
        os.mkdir(self.folder)
        pdf_images = convert_from_path(path, dpi=self.dpi)
        for idx, img in (enumerate(pdf_images)):
            self._translate_a_page(img, num=idx)

    def marge_pdf(self, file_name):
        pdf_file = glob.glob(f"{self.folder}/*.pdf")
        #pdf_file.sort()
        marger = pypdf.PdfMerger()
        for p in pdf_file:
            marger.append(p)
        marger.write(f"{file_name[:-4]}_ja.pdf")
        marger.close()
        shutil.rmtree(self.folder)
    
    def run(self, pdf_path):
        if pdf_path[-4:] != ".pdf":
            raise Exception("pdfを選んでね")
        file_name = os.path.basename(pdf_path)
        self.translate_pdf(pdf_path)
        self.marge_pdf(file_name)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("arg1")
    args = parser.parse_args()
    translater = PdfTranslate()
    translater.run(args.arg1)



