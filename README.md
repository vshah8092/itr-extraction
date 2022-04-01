# ITR Extraction
Apply ML techniques to extract entities from ITR (Income Tax Returns) documents.

## Steps:-
* Add the files to be processed in the [data/dataset](data/dataset/) folder.
* Download the custom-trained YOLOv5 model weights from [here](https://drive.google.com/file/d/1owvRRyQTRkmejrSDaXDR26chJ13UUxFZ/view?usp=sharing), and add the file in [src/yolov5](src/yolov5/) folder.
* Run [runner.py](src/runner.py) in the [src](src/) folder.
* The outputs ([annotated images](data/annotated_images/), [JSON files](data/json_files/), [output Excel/CSV](data/output_excel/)) are stored in the releavnt [data](data/) folders.