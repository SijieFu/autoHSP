# autoHSP - Autonomous Determination of Hansen Solubility Parameters via Active Learning

## Table of Contents
- [autoHSP - Autonomous Determination of Hansen Solubility Parameters via Active Learning](#autohsp---autonomous-determination-of-hansen-solubility-parameters-via-active-learning)
  - [Table of Contents](#table-of-contents)
  - [Citation](#citation)
  - [Overview](#overview)
  - [Structure](#structure)
  - [Installation](#installation)
  - [Deployment](#deployment)
    - [Streamlit app](#streamlit-app)
    - [Flask app](#flask-app)
    - [(Optional) Interface detection API](#optional-interface-detection-api)
    - [(Optional) Nginx - Reverse proxy](#optional-nginx---reverse-proxy)
  - [Usage](#usage)
    - [Streamlit app](#streamlit-app-1)
    - [Flask app](#flask-app-1)
    - [Solvent selection](#solvent-selection)
    - [Semi-automated lab experiments](#semi-automated-lab-experiments)
    - [Interface detection](#interface-detection)
  - [License](#license)
  - [Acknowledgements](#acknowledgements)

## Citation
This project is a part of the *autoHSP* preprint/publication:
```
@misc{fu_autonomous_2025,
  title={Autonomous Determination of Hansen Solubility Parameters via Active Learning},
  authors={Fu, Sijie and Wang, Daniel and Henderiks, Harmanna and Nogueira Assis, Andre and Charron, James and Washburn, Newell},
  date={2025-08-06},
  doi={10.26434/chemrxiv-2025-c7g4r},
  url = {https://chemrxiv.org/engage/chemrxiv/article-details/685b02eb3ba0887c335cf804},
}
```

## Overview
This project is designed to leverage a remote lab for autonomous research. More specifically, `autoHSP` uses a remote lab and generates the corresponsing experimental protocols for (semi-)automated experimentation without needing in-house setup.

*autoHSP* generates bi-solvent mixtures for measuring Hansen Solubility Parameters (HSPs), which greatly expands the solvent library and search space. Thus, it selects solvents, schedule experiments, and analyzes the results in a closed-loop and autonomous fashion. *autoHSP* does not perform an exhaustive search over the entire solvent space, but rather intelligently selects solvents based on the results of previous experiments. *autoHSP* implements a computer vision (CV) algorithm to analyze the images of the vials and determine the misciblity, as well as a batch-model active learning (BMAL) algorithm to select the solvents and schedule the experiments.

This project uses [Flask](https://flask.palletsprojects.com/) as the backend API to host *autoHSP* and [Streamlit](https://streamlit.io/) as the frontend for visualizing and interacting with the data and results. Streamlit also handles some backend logic for data manipulation.

## Structure
```bash
. # root directory of the autoHSP project
├── .streamlit/
│   ├── config.toml # Streamlit configuration file
│   └── default_login.yaml # default login configuration file
├── data/
│   ├── exp_config/ # directory for experiment configurations
│   │   ├── default.toml # default config for a test resin
│   │   ├── {resin_code}.toml # resin-specific config
│   │   └── readme.md
│   ├── exp_record/ # directory for experiment records
│   │   ├── {resin_code}_prior.csv # prior info about a resin
│   │   ├── {resin_code}.csv # resin-specific records
│   │   └── ...
│   ├── solvents.csv # solvents for the experiments
│   ├── resins.csv # resins for the experiments
│   └── ...
├── HSP/ # core HSP module for solvent selection and experiment scheduling
│   ├── api.py # API for the Flask app to handle experiment scheduling
│   ├── hsp_{...}.py # solvent selection module (irrelevant to experiment scheduling)
│   ├── info.py # configuration
│   ├── tasks.py # more detailed implementation of the experiment scheduling
│   └── {...}.py # scripts for experiment scheduling (related to experiment scheduling)
├── images/ # needs extra download step
│   ├── analysis/ # directory for analysis results
│   │   ├── {md5}.json # analysis results for the image with MD5 hash
│   ├── {md5}.jpg # collected images in autoHSP
│   └── readme.json # a mapping between MD5 and image names
├── resources/  #  additional resources/information
│   ├── autoHSP_lab_script.nb # Mathematica notebook for the lab-specific script
│   ├── autoHSP_lab_script.pdf # PDF version of the Mathematica notebook
│   └── autoHSP_nginx_conf # Nginx config for `autoHSP`
├── st_app_funcs/
│   └── {...}.py # different pages for the Streamlit app
├── templates/ # static HTML files for the Flask app
├── .gitignore
├── download.py # script to download additional resources (images, etc.)
├── Flask_test.py # testing Flask APIs
├── Flask_utils.py # utility functions for Flask app
├── Flask.py # Flask app
├── info.py # config shared between Streamlit and Flask apps
├── LICENSE
├── login.py # login functions for Streamlit app
├── main.py # Streamlit app
├── README.md # this file
├── requirements.txt # Python dependencies
├── selection_example.py # test for the solvent selection algorithm
└── utils.py # utility functions for Flask and Streamlit
```

## Installation
Python 3.12 is recommended for this project. The following instructions assume you have `conda` installed. You can also use other Python package managers.

1. Clone the repository:
    ```bash
    git clone https://github.com/SijieFu/autoHSP.git
    cd autoHSP
    ```
2. Create a new conda environment:
    ```bash
    conda create -n autoHSP python==3.12.9 -y
    conda activate autoHSP
    ```
3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
4. (Optional) Download the `autoHSP` images and analysis results:
    ```bash
    pip install gdown==5.2.0
    python download.py --all # this will download all resources
    ```
    If you have previously downloaded the resources and extracted them, you can skip this step. Otherwise, you may need to use `python download.py --all --overwrite` to overwrite the existing files.

## Deployment
The [run.sh](run.sh) script provides a simple way to run [the Streamlit app](#streamlit-app) and the [Flask app](#flask-app) together. You can start the apps with the following command:
```bash
bash run.sh # or `./run.sh` if you have executable permission
```
This will start the Streamlit app on port 8503 and the Flask app on port 5000 by default. To terminate the apps, run `bash run.sh --kill` or `./run.sh --kill`. You can also run invidual apps separately if you prefer. Check the followiong sections on the [Streamlit app](#streamlit-app) and the [Flask app](#flask-app) for more details.

### Streamlit app
As introduced in the [publication](#citation), the Streamlit app is designed to visualize the data results. It also offers a user-friendly interface to interact with the results and correct them if necessary.

Since the Streamlit app is designed for remote access, it is protected by a login system. The default login credentials are stored in [.streamlit/default_login.yaml](.streamlit/default_login.yaml). By default, an admin account is set up with the username `admin` and the password `not4production`. You should **NOT** use the default login credentials in production. Instead,
1. Make a copy first:
    ```bash
    cp .streamlit/default_login.yaml .streamlit/login.yaml
    ```
2. Edit the `.streamlit/login.yaml` file to set your own login credentials. You **must** set the `cookie.key` to a random string and change the information in the `credentials` section. You should also change `cookie.name`.
    > You can type your password in plain text and it will be hashed automatically when you run the Streamlit app. **However**, plaint text passwords are **never** recommended. **Use at your own risk**. If you are only serving the app locally, see the next step to disable email verification and use the app itself for registration/password reset.
3. (Recommended) If you want to enable email verification, get an API key from [StAuthenticator](https://stauthenticator.com/) and set it in the `api_key` field in the [.streamlit/login.yaml](.streamlit/login.yaml) file. This is not required for the app to run, but you can use email-based login and password reset features.
    > You **MUST** set the `api_key` field if you plan to serve the app on a public domain. Otherwise, anyone can access the app by resetting the password of a known user. The default login credentials has a fake `api_key` set to enable email verification by default. To disable email verification, set `api_key: null` in the `.streamlit/login.yaml` file or set `ENABLE_TWO_FACTOR_AUTH = False` in [info.py](info.py).
4. (Optional) Once you have set up the email verification, you can `pre-authorize` users to register by adding their email addresses to the `pre-authorized.emails` section in the `.streamlit/login.yaml` file.
5. (Optional) For invited guests, you can set up `guest-credentials` in the `.streamlit/login.yaml` file. This will allow users to bypass the login system by accessing `{domain}/{path}?login=guest-{guest_pass}`. For pass-only guest login, turn OAuth2 off explicitly by setting `oauth2: false` for the corresponding guest credential.
6. (Optional) If you want to use OAuth2 for guest login, you can set up `guest-credentials` first, and define the OAuth2 provider(s) in the `oauth2` section.

You can now run the Streamlit app with the following command:
```bash
streamlit run main.py
```

The Streamlit configuration file is located at `.streamlit/config.toml`. Once you run the Streamlit app, you can access it at [http://localhost:8503/HSP](http://localhost:8503/HSP) by default. If you have [Nginx](#optional-nginx---reverse-proxy) configured, you can access it at `{domain}/HSP`.

### Flask app
The Flask app is designed to host API endpoints for the *autoHSP* lab experiments. It is not designed to be accessed directly by users, but rather by the Streamlit app and through HTTP requests. However, you can still start the Flask app with the following command:
```bash
python Flask.py
```
The Flask app will run on [http://localhost:5000](http://localhost:5000) by default. If you have [Nginx](#optional-nginx---reverse-proxy) configured, you can access it at `{domain}/api/{API_KEY_A}`.

The Flask app answers the following API endpoints:
```plaintext
POST /next (for getting the next experiments for a test resin)
POST /upload (for uploading image for a sample)
POST /notify (for notifying the server of a success/exception/...)
POST /test (for testing the API)
```

You can test the experimental scheduling algorithm by accessing the `/test` endpoint, one of:
- [http://localhost:5000/test/](http://localhost:5000/test/)
- `{domain}/HSP-demo/` if you have [Nginx](#optional-nginx---reverse-proxy) configured.

The test will use the default test resin `R0` and the default solvent library. You can inspect the results in the Streamlit app at <http://localhost:8503/HSP/?task=View+past%2Fcurrent+experimental+records&resin=R0>, or `{domain}/HSP/?task=View+past%2Fcurrent+experimental+records&resin=R0` if you have [Nginx](#optional-nginx---reverse-proxy) configured. Use the key `R` on your keyboard to refresh the Streamlit page if needed.

### (Optional) Interface detection API
The Streamlit app has a playground page for testing the interface detection algorithm, but the algorithm is not included in this repository. The `/upload` endpoint of the Flask app is also dependent on the external interface detection API for image analysis.

You can set up the interface detection API (also a Flask app) by following the instructions in the repository: [SijieFu/interface-detection](https://github.com/SijieFu/interface-detection). Once setup, you can access the interface detection API at <http://localhost:5001> by default. If you have [Nginx](#optional-nginx---reverse-proxy) configured, you can access it at `{domain}/api/{API_KEY_B}`.

### (Optional) Nginx - Reverse proxy
The [Nginx configuration](resources/autoHSP_nginx_conf) is provided in [resources/autoHSP_nginx_conf](resources/autoHSP_nginx_conf). It is provided as a reference and you must modify it if you plan to use it, e.g., `API_KEY_A` and `API_KEY_B`. The proxy pass defaults are consistent with the Flask app and the Streamlit app defaults.

## Usage
If you are trying to understand how `autoHSP` works or apply it to your own research, here are the major components:

### Streamlit app
- The entry point for the Streamlit app is [main.py](main.py). It is the main script that runs the Streamlit app. The configuration file is located at [.streamlit/config.toml](.streamlit/config.toml).
- [login.py](login.py) contains the login functions for the Streamlit app. Copy the default login configuration file from [.streamlit/default_login.yaml](.streamlit/default_login.yaml) to [.streamlit/login.yaml](.streamlit/login.yaml) and edit it to set your own login credentials. See the [deployment](#streamlit-app) section for more details.
- The [st_app_funcs/](st_app_funcs/) directory contains different pages/functions for the Streamlit app. You can click around the Streamlit app and check the code in this directory to see how the app works.
- [info.py](info.py) contains the shared configuration for the Streamlit app and the Flask app.

### Flask app
- The entry point for the Flask app is [Flask.py](Flask.py). It is the main script that runs the Flask app.
- [Flask_test.py](Flask_test.py) contains the test API endpoints for the Flask app. A virtual test resin `R0` is reserved for testing purposes. Access the `/test` endpoint at <http://localhost:5000/test/>, or `{domain}/HSP-demo/` if you have [Nginx](#optional-nginx---reverse-proxy) configured, to test the experimental scheduling algorithm.
- [Flask_utils.py](Flask_utils.py) and [templates](templates) contain utility functions and HTML templates for the Flask app.
- For solvent selection/experiment scheduling, check [Flask.next](Flask.py). It uses the [HSP](HSP) module to get the next experiments for a resin.
- For handling image uploads from the designated lab, check [Flask.upload](Flask.py). It connects to the interface detection API to analyze the images and store the results.
- [info.py](info.py) contains the shared configuration for the Streamlit app and the Flask app.

### Solvent selection
- The `hsp_{...}.py` scripts in the [HSP](HSP) directory contain the solvent selection algorithms. [HSP/info.py](HSP/info.py) contains the configuration for the solvent selection algorithms (and the designated lab experiments).
- The solvent selection algorithm is for general use. You define the solvent library and the mixing strategy (see [HSP.hsp_utils.get_possible_solvents](HSP/hsp_utils.py) for details). With all possible candidates, you can use the [HSP.hsp_solver.propose_new_solvents](HSP/hsp_solver.py) function to get the selections, given any prior information about `compatible`, `incompatible`, and `pending` solvents.
- The selection works beyond 3D space - it is general across any number of dimensions (not tested in 1D though). Check [HSP.info.info.HSP_LENGTH](HSP/info.py) and [HSP.info.info.ENABLE_HSP_DISTANCE](HSP/info.py) for customization.
- The [selection_example.py](selection_example.py) script offers a simple example of how to use the solvent selection algorithm. To replicate the results shown in the [publication](#citation), you can run the script with the following command:
    ```bash
    python selection_example.py # for the default distance metric
    ```
    or
    ```bash
    TEST_AGG_DIST=1 python selection_example.py # for the more aggressive distance metric
    ```

### Semi-automated lab experiments
For experiment scheduling, the [HSP](HSP) directory contains the scripts for scheduling experiments (and the above [solvent selection](#solvent-selection)).

- [HSP/api.py](HSP/api.py), [HSP/tasks.py](HSP/tasks.py), and [HSP/utils.py](HSP/utils.py) are the main scripts.
- [HSP.api.get_tasks_for_thread](HSP/api.py) is the wrapper function for the Flask API `/next` endpoint. It returns the next experiments for a requested resin. You can start from here to understand how the lab experiments are scheduled.
- [data/exp_config/default.toml](data/exp_config/default.toml) is the default configuration file for a resin. It contains the default parameters for guiding the experimentation. Each resin/test material requires a specific configuration file, stored under the same directory - you **MUST** ensure that the configuration is legit for every test resin, such that no hazzardous experiments can be scheduled.
- [HSP/utils.py](HSP/utils.py) has many utility functions. For example, [HSP.utils.init_resin_config](HSP/utils.py) helps run the above setup.

### Interface detection
The interface detection algorithm is not included in this repository. However, you can check [Flask.upload](Flask.py) for how to connect to the interface detection API and get the analysis results. The playground page in the Streamlit app at <http://localhost:8503/HSP/?task=Playground> also provides a simple interface for testing the interface detection algorithm.

The interface detection API is a separate project, which you can find in the repository: [SijieFu/interface-detection](https://github.com/SijieFu/interface-detection).

## License
This project is licensed under [Apache-2.0](LICENSE). If you plan to use the interface detection algorithm, check [SijieFu/interface-detection](https://github.com/SijieFu/interface-detection) for its license and usage.

## Acknowledgements
This project is funded by [Covestro](https://www.covestro.com/). The CMU Bakery Square Lab also provided support for the lab experiments.

Special thanks to Florent Letronne and Benjamin Kline for their assistance in debugging and refining the laboratory experiment script and procedures.