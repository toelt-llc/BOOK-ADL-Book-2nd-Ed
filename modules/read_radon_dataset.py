import os
import pandas as pd
import numpy as np
from six.moves import urllib

import tensorflow as tf


class read_data():

  def __init__(self, CACHE_DIR, url_base):
    # We'll use the following directory to store files we download as well as our
    # preprocessed dataset.
    self.CACHE_DIR = CACHE_DIR
    self.url_base = url_base

  def cache_or_download_file(self, cache_dir, url_base, filename):
    """Read a cached file or download it."""
    filepath = os.path.join(cache_dir, filename)
    if tf.io.gfile.exists(filepath):
      return filepath
    if not tf.io.gfile.exists(cache_dir):
      tf.io.gfile.makedirs(cache_dir)
    url = os.path.join(url_base, filename)
    print('Downloading {url} to {filepath}.'.format(url=url, filepath=filepath))
    urllib.request.urlretrieve(url, filepath)
    return filepath


  def download_radon_dataset(self):
    """Download the radon dataset and read as Pandas dataframe."""
    srrs2 = pd.read_csv(self.cache_or_download_file(self.CACHE_DIR, self.url_base, 'srrs2.dat'))
    srrs2.rename(columns = str.strip, inplace = True)
    cty = pd.read_csv(self.cache_or_download_file(self.CACHE_DIR, self.url_base, 'cty.dat'))
    cty.rename(columns = str.strip, inplace = True)
    return srrs2, cty


  def preprocess_radon_dataset(self, srrs2, cty, state = 'MN'):
    """Preprocess radon dataset as done in "Bayesian Data Analysis" book."""
    srrs2 = srrs2[srrs2.state == state].copy()
    cty = cty[cty.st == state].copy()

    # We will now join datasets on Federal Information Processing Standards
    # (FIPS) id, ie, codes that link geographic units, counties and county
    # equivalents. http://jeffgill.org/Teaching/rpqm_9.pdf
    srrs2['fips'] = 1000 * srrs2.stfips + srrs2.cntyfips
    cty['fips'] = 1000 * cty.stfips + cty.ctfips

    df = srrs2.merge(cty[['fips', 'Uppm']], on='fips')
    df = df.drop_duplicates(subset = 'idnum')
    df = df.rename(index = str, columns = {'Uppm': 'uranium_ppm'})

    # For any missing or invalid activity readings, we'll use a value of `0.1`.
    df['radon'] = df.activity.apply(lambda x: x if x > 0. else 0.1)

    # Remap categories to start from 0 and end at max(category).
    county_name = sorted(df.county.unique())
    df['county'] = df.county.astype(
        pd.api.types.CategoricalDtype(categories = county_name)).cat.codes
    county_name = list(map(str.strip, county_name))

    df['log_uranium_ppm'] = df['uranium_ppm']
    df_features = df[['floor', 'county', 'log_uranium_ppm', 'pcterr']]
    df_labels = df['radon']

    return df_features, df_labels, county_name


  def create_dataset(self):
    """Return the final dataframe with preprocessed data."""
    radon_features, radon_labels, county_name = self.preprocess_radon_dataset(*self.download_radon_dataset())

    return radon_features, radon_labels, county_name
