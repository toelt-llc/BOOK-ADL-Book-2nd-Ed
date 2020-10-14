# ADL-Book-2nd-Ed
Github repository for the second edition of the ADL Book.

## Jupyter Book

To get all notebooks together in an easy online version we use [Jupyter Book](https://jupyterbook.org/intro.html). In this way the entire set of notebooks will become a browasable online book that will be accessible to everyone.

On the top right of each page you will always find several icons that will allow you to open a notebook in google colab, go full screen,  access the github repository or download the file.

![jupyter book icons](https://github.com/toelt-llc/ADL-Book-2nd-Ed/blob/master/images/jupyterbook-icon2.png)

## Using Jupyter Notebook (for authors of the book)

If you are contributing to the book and in particular to the online version you need to install jupyter notebook. I suggest the following steps

    virtualenv juypter-book
    source jupyter-book/bin/activate
    pip install -U jupyter-book
    
**Note:** when you create the jupyter-book environment do it somewhere since the command will create a directory that you don't want to have around. But you should know that. In case you are new to this please study the [official documentation on virtualenv on the official python website](https://docs.python.org/3/tutorial/venv.html).
