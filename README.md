### How do I get set up? ###

Prepare to part ways with Windows.

Get a Linux working environment (e.g. Ubuntu 14.04 LTS).

Open your terminal.

Execute the following commands:

* sudo apt-get update
* sudo apt-get install git python python-pip


The following three steps can be done in parallel:

* sudo apt-get install build-essential libopencv-dev python-opencv libboost-all-dev g++ python-numpy python-opencv python-pygame python-scipy python-matplotlib
* clone the repository (git clone https://github.com/Phazyck/Thesis.git)
* sudo pip install futures networkx cv2 pymunk tools


Once all three steps are done:

* cd Thesis
* sudo python setup.py install

You can now take a short break to marvel at all the hideous errors, spreading like a cancer in your terminal.

Remember kids: Real programmers wear wife-beaters and digitally abuse their compilers on a daily basis.
