from IPython.display import Image

class display_image():

    def __init__(self):
        # images' path
        self.img_path = 'ADL-Book-2nd-Ed/images/'

    def display_img(self, img, chapt):
        """Display an image present in the GitHub repository."""
        pil_img = Image(filename = self.img_path + chapt + '/' + img)
        display(pil_img)