# 2nd Edition of _Applied Deep Learning - A case based approach_
Github repository for the second edition of the ADL Book.

## Introduction

This repository is not the one that final readers will see. This contains only material that will be used to create the final online book that will be available to the readers. If you are contributing to the book the only thing you need to do is to create a notebook with google colab and upload it (when finished and approved) in the right folder. At the moment the folders contains also the word files of the chapters.

## Jupyter Book

To get all notebooks together in an easy online version we use [Jupyter Book](https://jupyterbook.org/intro.html). In this way the entire set of notebooks will become a browasable online book that will be accessible to everyone.

On the top right of each page you will always find several icons that will allow you to open a notebook in google colab, go full screen,  access the github repository or download the file.

![jupyter book icons](https://github.com/toelt-llc/ADL-Book-2nd-Ed/blob/master/images/jupyterbook-icon2.png)

## Using Jupyter Notebook (for authors of the book)

### Installation

If you are contributing to the book and in particular to the online version you need to install jupyter notebook. I suggest the following steps

    virtualenv juypter-book
    source jupyter-book/bin/activate
    pip install -U jupyter-book
    
**Note:** when you create the jupyter-book environment do it somewhere since the command will create a directory that you don't want to have around. But you should know that. In case you are new to this please study the [official documentation on virtualenv on the official python website](https://docs.python.org/3/tutorial/venv.html).

### Cloning of the repository

Now you should clone this repository

    git clone https://github.com/toelt-llc/ADL-Book-2nd-Ed.git
    
At the moment there is no automatic way of rebuilding the book. Is a manual process that not everyone should do. But if you want to chekc the book you will find it under the folder ```jupyterbook/_build/html/landingpage.html```.
