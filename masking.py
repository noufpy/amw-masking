from PIL import Image, ImageDraw, ImageFilter
import os
import cv2

directory = 'original'
maskType = 'u2net'
size = 256, 256

model = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'
)

for filename in os.listdir(directory):
    print(filename)
    f = os.path.join(directory, filename)
    basename = os.path.splitext(filename)[0]
    # im1 = Image.open(f)
    im = cv2.imread(f)

    # Convert image to grey scale
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = model.detectMultiScale(gray, 1.1, 4)

    # Blue Face Region
    for i, (x, y, w, h) in enumerate(faces):
        # Face rectangle with 2 points
        face = im[y:y+h, x:x+w]
        face = cv2.blur(face, (50,50))
        im[y:y+h, x:x+w] = face

    # Prep for masking
    color_coverted = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im1 = Image.fromarray(color_coverted)
    im2 = Image.open('White-Cover.jpg').resize(im1.size)
    mask = Image.open(maskType + '/' + basename + '.png').convert('L')

    # Composite three images together (image1 - original, image2 - white cover, mask)
    im = Image.composite(im1, im2, mask)

    # resize with aspect ratio maintained
    im.thumbnail(size, Image.ANTIALIAS)

    # paste resizing on 256x256 white background and center it
    im3 = Image.open('White-Cover.jpg')
    width, height = im.size
    im3.paste(im, (int(128-(width/2)), int(128-(height/2))))
    
    im3.save("results-" + maskType + '/' + basename + ".png")