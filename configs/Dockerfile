################################################################################
# Args
################################################################################
ARG USERNAME=mluser
ARG UPASSWD=password
ARG GID=1000
ARG UID=1000

################################################################################
# Biuid Anaconda Base
################################################################################
FROM nvidia/cuda:11.0-cudnn8-devel-ubuntu18.04 AS conda

# We need to redefine ARG TENSORFLOW_VERSION etc... here to get the default ARG value defined above the FROM instruction.
# See: https://docs.docker.com/engine/reference/builder/#understand-how-arg-and-from-interact
ARG USERNAME
ARG UPASSWD
ARG GID
ARG UID
ENV _USERNAME=${USERNAME}

# install gosu
# RUN set -eux; \
# 	apt-get update; \
# 	apt-get install -y gosu sudo; \
# 	rm -rf /var/lib/apt/lists/*; \
#     # verify that the binary works
# 	gosu nobody true
RUN apt-get update && \
    apt-get install sudo && \
    apt-get autoremove -y && \
    apt-get clean

# Add user for anaconda installation
RUN groupadd  ${USERNAME} -g ${GID} && \
    chgrp -R ${USERNAME} /opt/ && \
    chmod 770 -R /opt/ && \
    echo ${USERNAME} ${GID} ${UPASSWD}&& \
    useradd -u ${UID} -g ${USERNAME} -m ${USERNAME} --create-home --shell /bin/bash && \
    echo "${USERNAME} ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} && \
    chmod 0440 /etc/sudoers.d/${USERNAME} && \
    echo "Set disable_coredump false" >> /etc/sudo.conf && \
    echo ${USERNAME}:${UPASSWD} | chpasswd

COPY --chown=${USERNAME}:${USERNAME} clean-layer.sh  /tmp/clean-layer.sh
COPY --chown=${USERNAME}:${USERNAME} ./install-list/apt-install-list.txt /tmp/apt-install-list.txt
RUN apt-get update && \
    apt-get install -y $(grep -vE "^\s*#" /tmp/apt-install-list.txt  | tr "\n" " ") && \
    apt-get autoremove -y && \
    apt-get clean

# Install OpenCL & libboost (required by LightGBM GPU version)
RUN apt-get install -y ocl-icd-libopencl1 clinfo libboost-all-dev && \
    mkdir -p /etc/OpenCL/vendors && \
    echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd && \
    apt-get autoremove -y && \
    apt-get clean

# install Nodejs
RUN curl -sL https://deb.nodesource.com/setup_12.x | bash - && \
    apt-get install -y nodejs && \
    apt-get autoremove -y && \
    apt-get clean

# Install Anaconda with non-root privilege
USER ${USERNAME}
ARG ANACONDA_INSTALLER_LOCATION=https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh \
WORKDIR /home/${USERNAME}
ENV PATH /opt/conda/bin:$PATH
RUN wget --quiet $ANACONDA_INSTALLER_LOCATION -O ~/anaconda.sh && \
    bash ~/anaconda.sh -b -p  /opt/conda && \
    rm ~/anaconda.sh && \
    echo "PATH=/opt/conda/bin/:$PATH" >> ~/.bashrc && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc && \
    find /opt/conda/ -follow -type f -name '*.a' -delete && \
    find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
    /opt/conda/bin/conda clean -afy && \
    /tmp/clean-layer.sh

# python version affects following solving enviconment in conda
RUN conda install python==3.7.10 && \
    /tmp/clean-layer.sh

################################################################################
# Biuid DL base
################################################################################
FROM conda as dl-base

ARG USERNAME
USER ${USERNAME}

# # Make sure the dynamic linker finds the right libstdc++
# ENV LD_LIBRARY_PATH=/opt/conda/lib
# b/128333087: Set PROJ_LIB to points to the proj4 cartographic library.
ENV CUDA_MAJOR_VERSION=11
ENV CUDA_MINOR_VERSION=0
ENV CUDA_VERSION=$CUDA_MAJOR_VERSION.$CUDA_MINOR_VERSION
LABEL com.nvidia.volumes.needed="nvidia_driver"
LABEL com.nvidia.cuda.version="${CUDA_VERSION}"
ENV LD_LIBRARY_PATH="/usr/local/nvidia/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
ENV PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/opt/bin:${PATH}
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Make sure the dynamic linker finds the right libstdc++
# ENV LD_LIBRARY_PATH=/opt/conda/lib
# without 'conda install -y anaconda', solving environment in the steps of rapids install fails. why?
RUN conda install -y anaconda && \
    /tmp/clean-layer.sh
RUN conda install -c rapidsai -c nvidia -c conda-forge -c anaconda -c defaults "rapids=21.06" "rapids-blazing=21.06" cudatoolkit=$CUDA_VERSION  && \
    /tmp/clean-layer.sh
RUN conda install -c pytorch "pytorch=1.7" "torchvision=0.8" "torchaudio=0.7" "torchtext=0.8" cudatoolkit=$CUDA_VERSION && \
    /tmp/clean-layer.sh
RUN conda install proj geos -c conda-forge  && \
    /tmp/clean-layer.sh
# b/128333086: Set PROJ_LIB to points to the proj4 cartographic library.
# ENV PROJ_LIB=/opt/conda/share/proj
RUN sudo apt install -y libgeos-dev libproj-dev && \
    /tmp/clean-layer.sh
RUN pip install cartopy && \
    /tmp/clean-layer.sh

# Install LightGBM with GPU
RUN sudo mkdir /usr/local/src/LightGBM && \
    sudo chown ${USERNAME}:${USERNAME} /usr/local/src/LightGBM
RUN cd /usr/local/src && \
    git clone --recursive https://github.com/microsoft/LightGBM && \
    cd LightGBM && \
    git checkout tags/v3.2.1 && \
    mkdir build && cd build && \
    cmake -DUSE_GPU=1 -DOpenCL_LIBRARY=/usr/local/cuda/lib64/libOpenCL.so -DOpenCL_INCLUDE_DIR=/usr/local/cuda/include/ .. && \
    make -j$(nproc) && \
    cd /usr/local/src/LightGBM/python-package && \
    python setup.py install --precompile && \
    sudo mkdir -p /etc/OpenCL/vendors && \
    sudo sh -c "echo 'libnvidia-opencl.so.1' > /etc/OpenCL/vendors/nvidia.icd" && \
    /tmp/clean-layer.sh
USER ${USERNAME}

# Install JAX (Keep JAX version in sync with CPU image)
RUN pip install jax==0.2.16 jaxlib==0.1.68+cuda$CUDA_MAJOR_VERSION$CUDA_MINOR_VERSION -f https://storage.googleapis.com/jax-releases/jax_releases.html && \
    /tmp/clean-layer.sh

# Reinstall packages with a separate version for GPU support.
RUN pip install tensorflow-gpu==2.4.1 && \
    pip install tensorflow-gcs-config==2.4.0 && \
    pip install tensorflow-addons==0.12.1 && \
    pip install mxnet-cu$CUDA_MAJOR_VERSION$CUDA_MINOR_VERSION && \
    /tmp/clean-layer.sh

# Install GPU-only packages
RUN pip install pycuda && \
    pip install pynvrtc && \
    # b/190622765 latest version is causing issue. nnabla fixed it in https://github.com/sony/nnabla/issues/892, waiting for new release before we can remove this pin.
    pip install pynvml==8.0.4 && \
    pip install nnabla-ext-cuda$CUDA_MAJOR_VERSION$CUDA_MINOR_VERSION && \
    /tmp/clean-layer.sh

# Re-add TensorBoard Jupyter extension patch
# b/139212522 re-enable TensorBoard once solution for slowdown is implemented.
# ADD patches/tensorboard/notebook.py /opt/conda/lib/python3.7/site-packages/tensorboard/notebook.py

# Remove the CUDA stubs.
# ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH_NO_STUBS"

################################################################################
# Biuid ML base
################################################################################
FROM dl-base as ml-base

USER root
RUN mkdir /src/ && chmod 777 /src/

ARG USERNAME
USER ${USERNAME}


# The anaconda base image includes outdated versions of these packages. Update them to include the latest version.
RUN pip install seaborn python-dateutil dask && \
    pip install pyyaml joblib husl geopy ml_metrics mne pyshp && \
    pip install pandas && \
    # Install h2o from source.
    # Use `conda install -c h2oai h2o` once Python 3.7 version is released to conda.
    sudo apt-get install -y default-jre-headless && \
    pip install -f https://h2o-release.s3.amazonaws.com/h2o/latest_stable_Py.html h2o && \
    /tmp/clean-layer.sh

RUN sudo mkdir /usr/share/nltk_data && \
    sudo chown $USERNAME:$USERNAME /usr/share/nltk_data
RUN sudo apt-get install -y libfreetype6-dev && \
    sudo apt-get install -y libglib2.0-0 libxext6 libsm6 libxrender1 libfontconfig1 --fix-missing && \
    pip install gensim && \
    pip install textblob && \
    pip install wordcloud && \
    pip install xgboost && \
    # Pinned to match GPU version. Update version together.
    # pip install lightgbm==3.2.1 && \
    pip install pydot && \
    pip install keras && \
    pip install keras-tuner && \
    pip install flake8 && \
    # Pinned because it breaks theano test with the latest version (b/178107003).
    pip install theano-pymc==1.0.11 && \
    pip install python-Levenshtein && \
    pip install hep_ml && \
    # NLTK Project datasets
    mkdir -p /usr/share/nltk_data && \
    # NLTK Downloader no longer continues smoothly after an error, so we explicitly list
    # the corpuses that work
    # "yes | ..." answers yes to the retry prompt in case of an error. See b/133762095.
    yes | python -m nltk.downloader -d /usr/share/nltk_data abc alpino averaged_perceptron_tagger \
    basque_grammars biocreative_ppi bllip_wsj_no_aux \
    book_grammars brown brown_tei cess_cat cess_esp chat80 city_database cmudict \
    comtrans conll2000 conll2002 conll2007 crubadan dependency_treebank \
    europarl_raw floresta gazetteers genesis gutenberg \
    ieer inaugural indian jeita kimmo knbc large_grammars lin_thesaurus mac_morpho machado \
    masc_tagged maxent_ne_chunker maxent_treebank_pos_tagger moses_sample movie_reviews \
    mte_teip5 names nps_chat omw opinion_lexicon paradigms \
    pil pl196x porter_test ppattach problem_reports product_reviews_1 product_reviews_2 propbank \
    pros_cons ptb punkt qc reuters rslp rte sample_grammars semcor senseval sentence_polarity \
    sentiwordnet shakespeare sinica_treebank smultron snowball_data spanish_grammars \
    state_union stopwords subjectivity swadesh switchboard tagsets timit toolbox treebank \
    twitter_samples udhr2 udhr unicode_samples universal_tagset universal_treebanks_v20 \
    vader_lexicon verbnet webtext word2vec_sample wordnet wordnet_ic words ycoe && \
    # Stop-words
    pip install stop-words && \
    pip install scikit-image && \
    /tmp/clean-layer.sh

RUN pip install ibis-framework && \
    # pip install mxnet && \
    pip install gluonnlp && \
    pip install gluoncv && \
    /tmp/clean-layer.sh


RUN pip install scipy && \
    # b/176817038 avoid upgrade to 0.24 which is causing issues with hep-ml package.
    pip install scikit-learn==0.23.2 && \
    # HDF5 support
    pip install h5py && \
    pip install biopython && \
    # PUDB, for local debugging convenience
    pip install pudb && \
    pip install imbalanced-learn && \
    # Profiling and other utilities
    pip install line_profiler && \
    pip install orderedmultidict && \
    pip install smhasher && \
    pip install bokeh && \
    pip install numba && \
    pip install datashader && \
    # Boruta (python implementation)
    pip install Boruta && \
    sudo apt-get install -y graphviz && pip install graphviz && \
    # Pandoc is a dependency of deap
    sudo apt-get install -y pandoc && \
    pip install git+git://github.com/scikit-learn-contrib/py-earth.git@issue191 && \
    pip install essentia && \
    /tmp/clean-layer.sh

# vtk with dependencies
RUN sudo apt-get install -y libgl1-mesa-glx && \
    pip install vtk && \
    # xvfbwrapper with dependencies
    sudo apt-get install -y xvfb && \
    pip install xvfbwrapper && \
    /tmp/clean-layer.sh

RUN pip install mpld3 && \
    pip install gpxpy && \
    pip install arrow && \
    pip install nilearn && \
    pip install nibabel && \
    pip install pronouncing && \
    pip install markovify && \
    pip install imgaug && \
    pip install preprocessing && \
    pip install path.py && \
    pip install Geohash && \
    # https://github.com/vinsci/geohash/issues/4
    sed -i -- 's/geohash/.geohash/g' /opt/conda/lib/python3.7/site-packages/Geohash/__init__.py && \
    pip install deap && \
    pip install tpot && \
    pip install scikit-optimize && \
    pip install haversine && \
    pip install toolz cytoolz && \
    pip install plotly && \
    pip install hyperopt && \
    pip install fitter && \
    pip install langid && \
    # Delorean. Useful for dealing with datetime
    pip install delorean && \
    pip install trueskill && \
    # Useful data exploration libraries (for missing data and generating reports)
    pip install missingno && \
    pip install pandas-profiling && \
    pip install s2sphere && \
    pip install bayesian-optimization && \
    pip install matplotlib-venn && \
    # b/184083722 pyldavis >= 3.3 requires numpy >= 1.20.0 but TensorFlow 2.4.1 / 2.5.0 requires 1.19.2
    pip install pyldavis==3.2.2 && \
    pip install mlxtend && \
    pip install altair && \
    # b/183944405 pystan 3.x is not compatible with fbprophet.
    pip install pystan==2.19.1.1 && \
    pip install ImageHash && \
    pip install ecos && \
    pip install CVXcanon && \
    pip install pymc3 && \
    pip install imagecodecs && \
    pip install tifffile && \
    pip install spectral && \
    pip install descartes && \
    pip install geojson && \
    pip install pydicom && \
    pip install wavio && \
    pip install SimpleITK && \
    pip install hmmlearn && \
    pip install bayespy && \
    pip install gplearn && \
    pip install PyAstronomy && \
    pip install squarify && \
    pip install fuzzywuzzy && \
    pip install python-louvain && \
    pip install pyexcel-ods && \
    pip install sklearn-pandas && \
    pip install stemming && \
    pip install fbprophet && \
    pip install holoviews && \
    pip install geoviews && \
    pip install hypertools && \
    pip install py_stringsimjoin && \
    pip install mlens && \
    pip install scikit-multilearn && \
    pip install cleverhans && \
    pip install leven && \
    pip install catboost && \
    pip install lightfm && \
    pip install folium && \
    pip install scikit-plot && \
    # dipy requires the optional fury dependency for visualizations.
    pip install fury dipy && \
    pip install plotnine && \
    pip install scikit-surprise && \
    pip install pymongo && \
    pip install geoplot && \
    pip install eli5 && \
    pip install implicit && \
    pip install kaggle && \
    /tmp/clean-layer.sh

RUN pip install tensorpack && \   
    # Add google PAIR-code Facets
    cd /opt/ && git clone https://github.com/PAIR-code/facets && cd facets/ && jupyter nbextension install facets-dist/ --user && \
    export PYTHONPATH=$PYTHONPATH:/opt/facets/facets_overview/python/ && \
    pip install pycountry && \
    pip install iso3166 && \
    pip install pydash && \
    pip install kmodes --no-dependencies && \
    pip install librosa && \
    pip install polyglot && \
    pip install mmh3 && \
    pip install fbpca && \
    pip install sentencepiece && \
    pip install cufflinks && \
    pip install lime && \
    pip install memory_profiler && \
    /tmp/clean-layer.sh

# install cython & cysignals before pyfasttext
RUN pip install --upgrade cython && \
    pip install --upgrade cysignals && \
    pip install pyfasttext && \
    # ktext has an explicit dependency on Keras 2.2.4 which is not
    # compatible with TensorFlow 2.0 (support was added in Keras 2.3.0).
    # Add the package back once it is fixed upstream.
    # pip install ktext && \
    pip install fasttext && \
    sudo apt-get install -y libhunspell-dev && pip install hunspell && \
    pip install annoy && \
    pip install category_encoders && \
    # google-cloud-automl 2.0.0 introduced incompatible API changes, need to pin to 1.0.1
    pip install google-cloud-automl==1.0.1 && \
    pip install google-cloud-bigquery==2.2.0 && \
    pip install google-cloud-storage && \
    pip install google-cloud-translate==3.* && \
    pip install google-cloud-language==2.* && \
    pip install google-cloud-videointelligence==2.* && \
    pip install google-cloud-vision==2.* && \
    # b/183041606#comment5: the Kaggle data proxy doesn't support these APIs. If the library is missing, it falls back to using a regular BigQuery query to fetch data.
    pip uninstall -y google-cloud-bigquery-storage && \
    # After launch this should be installed from pip
    pip install git+https://github.com/googleapis/python-aiplatform.git@mb-release && \ 
    pip install ortools && \
    pip install scattertext && \
    # Pandas data reader
    pip install pandas-datareader && \
    pip install wordsegment && \
    pip install wordbatch && \
    pip install emoji && \
    # Add Japanese morphological analysis engine
    pip install janome && \
    pip install wfdb && \
    pip install vecstack && \
    # yellowbrick machine learning visualization library
    pip install yellowbrick && \
    pip install mlcrate && \
    /tmp/clean-layer.sh

RUN pip install bleach && \
    pip install certifi && \
    pip install cycler && \
    pip install decorator && \
    pip install entrypoints && \
    pip install html5lib && \
    pip install ipykernel && \
    pip install ipython && \
    pip install ipython-genutils && \
    pip install ipywidgets && \
    pip install isoweek && \
    pip install jedi && \
    pip install Jinja2 && \
    pip install jsonschema && \
    pip install jupyter-client && \
    pip install jupyter-console && \
    pip install jupyter-core && \
    pip install MarkupSafe && \
    pip install mistune && \
    pip install nbconvert && \
    pip install nbformat && \
    pip install notebook && \
    pip install papermill && \
    pip install olefile && \
    pip install kornia && \
    pip install pandas_summary && \
    pip install pandocfilters && \
    pip install pexpect && \
    pip install pickleshare && \
    pip install Pillow && \
    # Install openslide and its python binding
    sudo apt-get install -y openslide-tools && \
    pip install openslide-python && \
    pip install ptyprocess && \
    pip install Pygments && \
    pip install pyparsing && \
    pip install pytz && \
    pip install PyYAML && \
    pip install pyzmq && \
    pip install qtconsole && \
    pip install six && \
    pip install terminado && \
    pip install tornado && \
    pip install tqdm && \
    pip install traitlets && \
    pip install wcwidth && \
    pip install webencodings && \
    pip install widgetsnbextension && \
    pip install pyarrow && \
    pip install feather-format && \
    # fastai >= 2.3.1 upgrades pytorch/torchvision. upgrade of pytorch will be handled in b/181966788
    pip install fastai==2.2.7 && \
    pip install allennlp && \
    # https://b.corp.google.com/issues/184685619#comment9: 3.9.0 is causing a major performance degradation with spacy 2.3.5
    pip install importlib-metadata==3.4.0 && \
    python -m spacy download en_core_web_sm && python -m spacy download en_core_web_lg && \
    sudo apt-get install -y ffmpeg && \
    /tmp/clean-layer.sh

    ###########
    #
    #      NEW CONTRIBUTORS:
    # Please add new pip/apt installs in this block. Don't forget a "&& \" at the end
    # of all non-final lines. Thanks!
    #
    ###########

RUN pip install flashtext && \
    pip install wandb && \
    pip install marisa-trie && \
    pip install pyemd && \
    pip install pyupset && \
    pip install pympler && \
    pip install s3fs && \
    pip install featuretools && \
    pip install -e git+https://github.com/SohierDane/BigQuery_Helper#egg=bq_helper && \
    pip install hpsklearn && \
    pip install git+https://github.com/Kaggle/learntools && \
    pip install kmapper && \
    pip install shap && \
    pip install ray && \
    pip install gym && \
    pip install pyarabic && \
    pip install pandasql && \
    pip install tensorflow_hub && \
    pip install jieba  && \
    pip install git+https://github.com/SauceCat/PDPbox && \
    # ggplot is broken and main repo does not merge and release https://github.com/yhat/ggpy/pull/668
    pip install https://github.com/hbasria/ggpy/archive/0.11.5.zip && \
    pip install cesium && \
    pip install rgf_python && \
    # b/185992410: onnx is a dependency of pytext, but the version 1.9.0 breaks pytext test.
    # Remove this installation when pytext fixes the problem.
    pip install onnx==1.8.1 && \
    # b/145404107: latest version force specific version of numpy and torch.
    pip install pytext-nlp==0.1.2 && \
    pip install tsfresh && \
    pip install pykalman && \
    pip install optuna && \
    pip install plotly_express && \
    pip install albumentations && \
    pip install catalyst && \
    pip install osmnx && \
    sudo apt-get -y install libspatialindex-dev && \
    pip install pytorch-ignite && \
    pip install qgrid && \
    pip install bqplot && \
    pip install earthengine-api && \
    pip install transformers && \
    pip install dlib && \
    pip install kaggle-environments && \
    pip install geopandas && \
    pip install nnabla && \
    pip install vowpalwabbit && \
    # papermill can replace nbconvert for executing notebooks
    pip install cloud-tpu-client && \
    # b/188429515#comment7 tensorflow-cloud >= 0.1.14 installs tensorflow-transform which install apache-beam which downgrades the google.cloud library to 1.x.
    pip install tensorflow-cloud==0.1.13 && \
    pip install tensorflow-datasets && \
    pip install pydub && \
    pip install pydegensac && \
    pip install pytorch-lightning && \
    pip install datatable && \
    pip install sympy && \
    # flask is used by agents in the simulation competitions.
    pip install flask && \
    # pycrypto is used by competitions team.
    pip install pycrypto && \
    pip install easyocr && \
    # Keep JAX version in sync with GPU image.
    # pip install jax==0.2.12 jaxlib==0.1.64 && \
    # ipympl adds interactive widget support for matplotlib
    pip install ipympl==0.7.0 && \
    pip install pandarallel && \
    /tmp/clean-layer.sh

# Download base easyocr models.
# https://github.com/JaidedAI/EasyOCR#usage
RUN mkdir -p home/${USERNAME}/.EasyOCR/model && \
    wget --no-verbose "https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/latin_g2.zip" -O home/${USERNAME}/.EasyOCR/model/latin.zip && \
    unzip /home/${USERNAME}/.EasyOCR/model/latin.zip -d /home/${USERNAME}/.EasyOCR/model/ && \
    rm /home/${USERNAME}/.EasyOCR/model/latin.zip && \
    wget --no-verbose "https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/english_g2.zip" -O /home/${USERNAME}/.EasyOCR/model/english.zip && \
    unzip /home/${USERNAME}/.EasyOCR/model/english.zip -d /home/${USERNAME}/.EasyOCR/model/ && \
    rm /home/${USERNAME}/.EasyOCR/model/english.zip && \
    wget --no-verbose "https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/craft_mlt_25k.zip" -O /home/${USERNAME}/.EasyOCR/model/craft_mlt_25k.zip && \
    unzip /home/${USERNAME}/.EasyOCR/model/craft_mlt_25k.zip -d /home/${USERNAME}/.EasyOCR/model/ && \
    rm /home/${USERNAME}/.EasyOCR/model/craft_mlt_25k.zip && \
    /tmp/clean-layer.sh

# Tesseract and some associated utility packages
RUN sudo apt-get install tesseract-ocr -y && \
    pip install pytesseract && \
    pip install wand && \
    pip install pdf2image && \
    pip install PyPDF && \
    pip install pyocr && \
    /tmp/clean-layer.sh
ENV TESSERACT_PATH=/usr/bin/tesseract

# For Facets
ENV PYTHONPATH=$PYTHONPATH:/opt/facets/facets_overview/python/
# For Theano with MKL
ENV MKL_THREADING_LAYER=GNU

# Temporary fixes and patches
    # Temporary patch for Dask getting downgraded, which breaks Keras
RUN pip install --upgrade dask && \
    # Stop jupyter nbconvert trying to rewrite its folder hierarchy
    mkdir -p /home/${USERNAME}/.jupyter && touch /home/${USERNAME}/.jupyter/jupyter_nbconvert_config.py && touch /home/${USERNAME}/.jupyter/migrated && \
    # mkdir -p /.jupyter && touch /.jupyter/jupyter_nbconvert_config.py && touch /.jupyter/migrated && \
    # Stop Matplotlib printing junk to the console on first load
    sed -i "s/^.*Matplotlib is building the font cache using fc-list.*$/# Warning removed by Kaggle/g" /opt/conda/lib/python3.7/site-packages/matplotlib/font_manager.py && \
    # Make matplotlib output in Jupyter notebooks display correctly
    # mkdir -p /etc/ipython/ && echo "c = get_config(); c.IPKernelApp.matplotlib = 'inline'" > /etc/ipython/ipython_config.py && \
    sudo mkdir -p /etc/ipython/ && sudo bash -c "echo \"c = get_config(); c.IPKernelApp.matplotlib = 'inline'\" > /etc/ipython/ipython_config.py" && \
    # Temporary patch for broken libpixman 0.38 in conda-forge, symlink to system libpixman 0.34 untile conda package gets updated to 0.38.5 or higher.
    sudo ln -sf /usr/lib/x86_64-linux-gnu/libpixman-1.so.0.34.0 /opt/conda/lib/libpixman-1.so.0.38.0 && \
    /tmp/clean-layer.sh

# gcloud SDK https://cloud.google.com/sdk/docs/quickstart-debian-ubuntu
RUN sudo bash -c "echo \"deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main\" \
    | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list" && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | \
    sudo apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - && \
    sudo apt-get update -y && sudo apt-get install google-cloud-sdk -y && \
    /tmp/clean-layer.sh

# Add BigQuery client proxy settings
ENV PYTHONUSERBASE "/${USERNAME}/.local"
COPY --chown=${USERNAME}:${USERNAME}  patches/kaggle_gcp.py /${USERNAME}/.local/lib/python3.7/site-packages/kaggle_gcp.py
COPY --chown=${USERNAME}:${USERNAME}  patches/kaggle_secrets.py /${USERNAME}/.local/lib/python3.7/site-packages/kaggle_secrets.py
COPY --chown=${USERNAME}:${USERNAME}  patches/kaggle_session.py /${USERNAME}/.local/lib/python3.7/site-packages/kaggle_session.py
COPY --chown=${USERNAME}:${USERNAME}  patches/kaggle_web_client.py /${USERNAME}/.local/lib/python3.7/site-packages/kaggle_web_client.py
COPY --chown=${USERNAME}:${USERNAME}  patches/kaggle_datasets.py /${USERNAME}/.local/lib/python3.7/site-packages/kaggle_datasets.py
COPY --chown=${USERNAME}:${USERNAME}  patches/log.py /${USERNAME}/.local/lib/python3.7/site-packages/log.py
COPY --chown=${USERNAME}:${USERNAME}  patches/sitecustomize.py /${USERNAME}/.local/lib/python3.7/site-packages/sitecustomize.py
# Override default imagemagick policies
COPY --chown=${USERNAME}:${USERNAME} patches/imagemagick-policy.xml /etc/ImageMagick-6/policy.xml
COPY --chown=${USERNAME}:${USERNAME} ./patches/nbconvert-extensions.tpl /opt/kaggle/nbconvert-extensions.tpl
COPY --chown=${USERNAME}:${USERNAME} ./patches/template_conf.json /opt/kaggle/conf.json

# TensorBoard Jupyter extension. Should be replaced with TensorBoard's provided magic once we have
# worker tunneling support in place.
# b/139212522 re-enable TensorBoard once solution for slowdown is implemented.
# ENV JUPYTER_CONFIG_DIR "/root/.jupyter/"
# RUN pip install jupyter_tensorboard && \
#     jupyter serverextension enable jupyter_tensorboard && \
#     jupyter tensorboard enable
# ADD patches/tensorboard/notebook.py /opt/conda/lib/python3.7/site-packages/tensorboard/notebook.py

# Disable unnecessary jupyter extensions
# RUN jupyter-nbextension disable nb_conda --py --sys-prefix && \
#     jupyter-serverextension disable nb_conda --py --sys-prefix && \
#     python -m nb_conda_kernels.install --disable

# Set backend for matplotlib
ENV MPLBACKEND "agg"

ARG GIT_COMMIT=unknown
ARG BUILD_DATE=unknown

LABEL git-commit=$GIT_COMMIT
LABEL build-date=$BUILD_DATE
LABEL tensorflow-version=$TENSORFLOW_VERSION
# Used in the Jenkins `Docker GPU Build` step to restrict the images being pruned.
LABEL kaggle-lang=python

# Correlate current release with the git hash inside the kernel editor by running `!cat /etc/git_commit`.
# RUN echo "$GIT_COMMIT" > /etc/git_commit && echo "$BUILD_DATE" > /etc/build_date

################################################################################
# Post Install Process
################################################################################
FROM ml-base as ml-env

ARG USERNAME
USER ${USERNAME}
WORKDIR /home/${USERNAME}

RUN mkdir -p ~/.local && \
    mkdir -p ~/workspace && \
    mkdir -p ~/.ssh && \
    mkdir -p ~/.kaggle && \
    mkdir -p ~/.aws

COPY --chown=${USERNAME}:${USERNAME} docker-entrypoint.sh /home/${USERNAME}/.local/bin/docker-entrypoint.sh
COPY --chown=${USERNAME}:${USERNAME} run-jupyter.sh /home/${USERNAME}/.local/bin/run-jupyter.sh
COPY --chown=${USERNAME}:${USERNAME} run-mlflow.sh /home/${USERNAME}/.local/bin/run-mlflow.sh
RUN echo "PATH=~/.local/bin/:$PATH" >> ~/.bashrc

RUN pip install mlflow && \
    /tmp/clean-layer.sh
RUN pip install jupyterlab==3.0 jupyterlab-lsp==3.8.0 jedi-language-server && \
    /tmp/clean-layer.sh
RUN pip install jupyterlab_vim && \
    /tmp/clean-layer.sh

USER root
RUN mkdir /var/run/sshd
