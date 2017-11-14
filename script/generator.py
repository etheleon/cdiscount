def myFastGenerator(length=100000):
    """
        file1 = open("/data/uesu/cdiscount/cdiscount-kernel/data/images_128x128.bin", 'rb')
        file1.seek(0)
        test = file1.read(7555)
        aimage = cv2.imdecode(np.frombuffer(test, dtype=np.uint8), cv2.IMREAD_COLOR)
        file1.seek(7555)
        test = file1.read(7853)
        bimage = cv2.imdecode(np.frombuffer(test, dtype=np.uint8), cv2.IMREAD_COLOR)
    """
    #labels = pd.read_csv("/data/uesu/cdiscount/cdiscount-kernel/data/images_128x128.csv").head(length)
    labels = feahter.read_dataframe("/data/uesu/cdiscount/data/meta.feather").head(length)
    with open("/data/uesu/cdiscount/cdiscount-kernel/data/images_128x128.bin", 'rb') as file_:
        # for class_, offset, size in list(examples)[start:end]:
        for startend in list(zip([start for start in range(1,length+1, 100)],[end for end in range(100,length+100, 100)])):
            examples = zip(
                labels['class'][startend[0]:startend[1]],
                labels['offset'][startend[0]:startend[1]],
                labels['size'][startend[0]:startend[1]]
                )
            tempArr = []
            for class_, offset, size in examples:
                file_.seek(offset)
                tempArr.append(cv2.imdecode(np.frombuffer(file_.read(size), dtype=np.uint8), cv2.IMREAD_COLOR))
            yield (np.concatenate([y[np.newaxis,:,:,:] for y in tempArr], axis=0), labels['class'][startend[0]:startend[1]].values)

def myFastGeneratorVal(length=100000):
    """
        file1 = open("/data/uesu/cdiscount/cdiscount-kernel/data/images_128x128.bin", 'rb')
        file1.seek(0)
        test = file1.read(7555)
        aimage = cv2.imdecode(np.frombuffer(test, dtype=np.uint8), cv2.IMREAD_COLOR)
        file1.seek(7555)
        test = file1.read(7853)
        bimage = cv2.imdecode(np.frombuffer(test, dtype=np.uint8), cv2.IMREAD_COLOR)
    """
    # labels = pd.read_csv("/data/uesu/cdiscount/cdiscount-kernel/data/images_128x128.csv").tail(length)
    labels = feahter.read_dataframe("/data/uesu/cdiscount/data/meta.feather").tail(length)
    with open("/data/uesu/cdiscount/cdiscount-kernel/data/images_128x128.bin", 'rb') as file_:
        # for class_, offset, size in list(examples)[start:end]:
        for startend in list(zip([start for start in range(1,length+1, 100)],[end for end in range(100,length+100, 100)])):
            examples = zip(
                labels['class'][startend[0]:startend[1]],
                labels['offset'][startend[0]:startend[1]],
                labels['size'][startend[0]:startend[1]]
                )
            tempArr = []
            for class_, offset, size in examples:
                file_.seek(offset)
                tempArr.append(cv2.imdecode(np.frombuffer(file_.read(size), dtype=np.uint8), cv2.IMREAD_COLOR))
            yield (np.concatenate([y[np.newaxis,:,:,:] for y in tempArr], axis=0), labels['class'][startend[0]:startend[1]].values)
