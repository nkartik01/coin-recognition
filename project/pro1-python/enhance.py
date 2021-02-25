import imutils


def stretch(img):
    # print(img[0])
    mi = 256
    ma = -1
    for i in img:
        for j in i:
            if j < mi:
                mi = j
            if j > ma:
                ma = j
    for i in range(len(img)):
        for j in range(len(img[0])):
            img[i][j] = (img[i][j]-mi)*255/(ma-mi)
    return(img)


def conservative_smoothing_gray(data, filter_size):

    temp = []

    indexer = filter_size // 2

    new_image = data.copy()

    nrow, ncol = data.shape

    for i in range(nrow):

        for j in range(ncol):

            for k in range(i-indexer, i+indexer+1):

                for m in range(j-indexer, j+indexer+1):

                    if (k > -1) and (k < nrow):

                        if (m > -1) and (m < ncol):

                            temp.append(data[k, m])

            temp.remove(data[i, j])

            max_value = max(temp)

            min_value = min(temp)

            if data[i, j] > max_value:

                new_image[i, j] = max_value

            elif data[i, j] < min_value:

                new_image[i, j] = min_value

            temp = []

    return new_image.copy()
