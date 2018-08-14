FROM ubuntu:14.04

# Install dependencies
RUN apt-get -qq update           &&  \
    apt-get -qq install --assume-yes \
        "build-essential"           \
        "cmake"                     \
        "git"                       \
        "wget"                      \
        "libopenjpeg2"              \
        "libopenblas-dev"           \
        "liblapack-dev"             \
        "libjpeg-dev"               \
        "libtiff5-dev"              \
        "zlib1g-dev"                \
        "libfreetype6-dev"          \
        "liblcms2-dev"              \
        "libwebp-dev"               \
        "gfortran"                  \
        "pkg-config"                \
        "python3"                   \
        "python3-dev"               \
        "python3-pip"               \
        "python3-numpy"             \
        "python3-scipy"             \
        "python3-six"               \
        "python3-networkx"       &&  \
    rm -rf /var/lib/apt/lists/*  &&  \
    python3 -m pip install "cython"

# Install requirements before copying project files
WORKDIR /nd
COPY requirements.txt .
RUN python3 -m pip install -r "requirements.txt"

# Copy only required project files
COPY doodle.py .

# Get a pre-trained neural network (VGG19)
RUN wget -q "https://github.com/alexjc/neural-doodle/releases/download/v0.0/vgg19_conv.pkl.bz2"

# Set an entrypoint to the main doodle.py script
ENTRYPOINT ["python3", "doodle.py", "--device=cpu"]
