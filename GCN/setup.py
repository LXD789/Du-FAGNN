from setuptools import setup
from setuptools import find_packages

setup(name='kegra',  # 生成的包名称
      version='0.0.1',  # 版本号
      description='Deep Learning on Graphs with Keras',  # 包的简要描述
      author='Thomas Kipf',  # 包的作者
      author_email='thomas.kipf@gmail.com',  # 包作者的邮箱地址
      url='https://tkipf.github.io',  # 程序的官网地址
      download_url='...',  # 程序的下载地址
      license='MIT',  # 程序的授权信息
      install_requires=['keras', 'scipy', 'numpy', 'tensorflow'],  # 需要安装的依赖包
      extras_require={  # 额外用于模型存储的依赖包
          'model_saving': ['json', 'h5py'],
      },
      package_data={'kegra': ['README.md']},
      # fine_packages()函数默认在和setup.py同一目录下搜索各个含有__init__.py的包
      packages=find_packages())
