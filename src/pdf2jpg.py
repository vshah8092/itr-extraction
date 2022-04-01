def pdf2jpg(path, name):
    import pdf2image
    
    from pdf2image import convert_from_path

    from config.constants import PathConstants

    images = convert_from_path(path, poppler_path = "")
    for i in range(len(images)):
        imagePath = PathConstants.DATASET_FILE_PATH + name + "page_" + str(i + 1) + ".jpg"
        images[i].save(imagePath, "JPEG")