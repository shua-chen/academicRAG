# import fitz
# import html2text
import os
import logging
import tqdm
import asyncio
import pipmaster as pm


from dataclasses import dataclass
from typing import List,AnyStr,Any
from .utils import always_get_an_event_loop
from tqdm.asyncio import tqdm as tqdm_asyncio

@dataclass
class DataPipe:
    encod: str="utf-8"
    def __post_init__(self):
        self.logger = logging.getLogger('data pipe')
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.info("DataPipe created")

    def load_input(self, path: str,return_paths: bool=False) -> List[str]|str:
        all_files=[]
        
        if os.path.isdir(path):
            for root,_,files in os.walk(path):
                for file in files:
                    full_path=os.path.join(root,file)
                    all_files.append(full_path)
        else:
            all_files.append(path)
        self.logger.info(f"Detecting {len(all_files)} files")

        extracted_texts=[]
        self.tmp_file_paths=[]
        
        
        loop=always_get_an_event_loop()
        tasks=[self.ahandle_single_file(file) for file in all_files]
        extracted_texts = loop.run_until_complete(
            tqdm_asyncio.gather(*tasks, desc="Extracting", total=len(tasks))
        )
        if len(extracted_texts)==1:
            extracted_texts=extracted_texts[0]
            self.tmp_file_paths=self.tmp_file_paths[0]
        if return_paths:
            return extracted_texts, self.tmp_file_paths
        else:
            return extracted_texts
    
    async def ahandle_single_file(self, path: str) -> List[str]:
        file_type = path.split(".")[-1]
        self.tmp_file_paths.append(path)
        try:
            if file_type == "pdf":
                return await self.handle_pdf(path)
            elif file_type=="html": 
                return await self.handle_html(path)
            elif file_type == "txt":
                return await self.handle_txt(path)
            elif file_type in ["json","jsonl"]:
                return await self.handle_json(path)
            else:
                return await self.handle_with_textract(path)
        except Exception as e:
            self.logger.error(f"Error while extracting text from {path}: {e}")
            self.tmp_file_paths.remove(path)
            return []
        
    async def handle_pdf(self, path: str) -> List[Any]:
        if not pm.is_installed("pymupdf"):
            pm.install("pymupdf")

        import fitz
        pdf_document = fitz.open(path)
        pdf_text = ""
        pdf_length=0
        for i,page in enumerate(pdf_document):
            page_text = page.get_text("text").strip()
            if page_text:
                pdf_text += page_text
                pdf_length+=len(page_text)
        pdf_document.close()

        if pdf_length==0:
            self.logger.warning(f"Cannot extract text from {path}, the file may be image-based PDF or empty.")
            return []
        return [pdf_text]
    
    async def handle_html(self, path: str) -> List[Any]:
        if not pm.is_installed("html2text"):
            pm.install("html2text")

        import html2text

        h = html2text.HTML2Text()
        h.ignore_links = True
        h.ignore_images = True
        with open(path, "r", encoding=self.encod) as f:
            html_text = h.handle(f.read().strip())

        if len(html_text)==0:
            self.logger.warning(f"Cannot extract text from {path}, the file may be empty.")
            return []
        return [html_text]
    
    async def handle_txt(self, path: str) -> List[Any]:
        with open(path, "r", encoding=self.encod) as f:
            txt_text = f.read().strip()
        
        if len(txt_text)==0:
            self.logger.warning(f"Cannot extract text from {path}, the file may be empty.")
            return []
        return [txt_text]
    
    async def handle_json(self, path: str) -> List[Any]:
        import json
        with open(path, "r", encoding=self.encod) as f:
            json_text = json.load(f)

        if len(json_text)==0:
            self.logger.warning(f"Cannot extract text from {path}, the file may be empty.")
            return []

        else:
            if isinstance(json_text, dict):
                json_text = json_text.values()
                json_text = [str(text).strip() for text in json_text]

            elif isinstance(json_text, list):
                json_text = [str(text).strip() for text in json_text]

            else:
                # type is string
                json_text = [json_text.strip()]
            return json_text
        
    async def handle_with_textract(self, path: str) -> List[Any]:
        if not pm.is_installed("textract"):
            raise Exception("Textract is not installed, please install it or convert your file to .pdf, .html, .txt or .json. format.")

        import textract
        text = textract.process(path).decode(self.encod)

        if len(text)==0:
            self.logger.warning(f"Cannot extract text from {path}, the file may be empty.")
            return []
        return [text]
            
            


        


