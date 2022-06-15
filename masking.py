from PIL import Image, ImageDraw, ImageFilter
import os
import cv2

directory = 'original'
model = 'u2net'
size = 256, 256

for filename in os.listdir(directory):
    print(filename)
    f = os.path.join(directory, filename)
    basename = os.path.splitext(filename)[0]
    im1 = Image.open(f)
    im2 = Image.open('White-Cover.jpg').resize(im1.size)

    mask = Image.open('masks-' + model + '/' + basename + '.png').convert('L')

    # Composite three images together (image1 - original, image2 - white cover, mask)
    im = Image.composite(im1, im2, mask)

    # resize with aspect ratio maintained
    im.thumbnail(size, Image.ANTIALIAS)

    # paste resizing on 256x256 white background and center it
    im3 = Image.open('White-Cover.jpg')
    width, height = im.size
    im3.paste(im, (int(128-(width/2)), int(128-(height/2))))
    
    im3.save("results-" + model + '/' + basename + ".png")