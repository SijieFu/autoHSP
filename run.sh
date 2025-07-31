#!/bin/bash
# a bash script to start the Flask app and/or the Streamlit app

# get the current directory
DIR=$(cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd)
DIRNAME=$(basename "$DIR")
cd "$DIR"

# force restart if `-f | --force` is given
# use key file to connect to GitHub if "-k | --key" is given
# use conda environment if "-e | --env" is given
# use server address if "-s | --server" is given
# do not run git operations if "--no-git" is given
# kill the services if "--kill | --shutdown" is given
FORCE=false
SSH_KEY_FILE="$HOME/.ssh/github-$DIRNAME"
CONDA_ENV_NAME="$DIRNAME"
SERVER_ADDRESS="localhost"
KILL_SERVICES=false
USAGE="Usage: $0 [-f|--force] [-k|--key <keyfile>] [-e|--env <conda_env_name>] \
[-s|--server <streamlit_server_address>] [--no-git] [--kill|--shutdown] [-h|--help]"
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -f|--force)
            FORCE=true
            shift
        ;;
        -k|--key)
            if [ -n "$2" ] && [[ "$2" != -* ]]; then
                SSH_KEY_FILE="$2"
                shift 2
            else
                echo "Error: -k|--key requires a file path argument."
                exit 1
            fi
        ;;
        -e|--env)
            if [ -n "$2" ] && [[ "$2" != -* ]]; then
                CONDA_ENV_NAME="$2"
                shift 2
            else
                echo "Error: -e|--env requires a conda environment name argument."
                exit 1
            fi
        ;;
        -s|--server)
            if [ -n "$2" ] && [[ "$2" != -* ]]; then
                SERVER_ADDRESS="$2"
                shift 2
            else
                echo "Error: -s|--server requires a server address argument."
                exit 1
            fi
        ;;
        --no-git)
            runGit=false
            shift
        ;;
        --kill|--shutdown)
            KILL_SERVICES=true
            shift
        ;;
        -h|--help)
            echo "$USAGE"
            exit 0
        ;;
        *)
            echo "Error: Invalid option '$key'."
            echo "$USAGE"
            exit 1
        ;;
    esac
done

# print the current date and time
cd "$DIR"
echo $(date +"%Y-%m-%d %H:%M:%S")
echo "The current directory is $DIR"

# activate the conda environment
echo "Activating the conda environment in $(which conda)"
#source conda.sh
source $(dirname $(which conda))/../etc/profile.d/conda.sh
conda activate "$CONDA_ENV_NAME" || {
    echo "Error: Failed to activate conda environment '$CONDA_ENV_NAME'."
    echo "If you are not using conda, make sure this shell script is run in a correct environment."
}
echo $(dirname $(which conda)) > /dev/null 2>&1

# Check if SSH_AUTH_SOCK is set and valid
if [ -z "$SSH_AUTH_SOCK" ] || ! pgrep -u "$USER" ssh-agent > /dev/null; then
    eval $(ssh-agent -s)
fi

# add key files
if [ ! -f "$SSH_KEY_FILE" ]; then
    echo "Error: SSH key file '$SSH_KEY_FILE' does not exist."
    echo "Make sure you have the credentials set up to access the git repository otherwise."
else
    echo "Using SSH key file: $SSH_KEY_FILE"
    ssh-add -l | grep -q "$(basename "$SSH_KEY_FILE")" || {
        echo "Adding SSH key file '$SSH_KEY_FILE' to the SSH agent."
        warning_msg="Warning: Failed to add SSH key '$SSH_KEY_FILE' to SSH agent.\nGit operations may fall back to other authentication methods."
        if [[ -t 0 ]]; then
            ssh-add "$SSH_KEY_FILE" || echo -e "$warning_msg"
        else
            echo "$SSH_PASSPHRASE" | SSH_ASKPASS_REQUIRE=force ssh-add "$SSH_KEY_FILE" || echo -e "$warning_msg"
        fi
    }
    export GIT_SSH_COMMAND="ssh -i '$SSH_KEY_FILE'"
fi

# check if git is installed and the current directory is a git repository
if ! [ "$runGit" = false ]; then
    echo "Running git operations..."
    if ! command -v git &> /dev/null; then
        echo "  Error: git is not installed. Ignoring git operations."
    else
        if ! git rev-parse --is-inside-work-tree &> /dev/null; then
            echo "  Warning: The current directory is not a git repository. Ignoring git operations."
        else
            git_pull_log=$(git pull 2>&1)
            echo "===== Git Pull Log ====="
            echo "$git_pull_log"

            # the last line of the git pull log indicates # of files changed
            branch_updated=$(echo "$git_pull_log" | tail -n 1 | grep -o "changed" | wc -l)
            if [ $branch_updated -gt 0 ]; then
                echo ">>> Branch has been updated. Services will be force restarted, if running."
                FORCE=true
            fi

            requirementsUpdated=$(echo "$git_pull_log" | grep "requirements.txt" | wc -l)
            if [ $requirementsUpdated -gt 0 ]; then
                echo ">>> The requirements.txt file has been updated. Installing the new packages"
                pip install -r requirements.txt > /dev/null 2>&1
            fi
            echo "===== End of Git Pull Log ====="
        fi
    fi
else
    echo "Skipping git operations as --no-git option is set."
fi


# check if the FORCESRESTART file exists as the force flag
if [ -e "FORCERESTART" ]; then
    FORCE=true
    echo "`FORCERESTART` file found. Forcing restart of the Streamlit app."
    rm FORCERESTART
fi


cd "$DIR"
echo "===== Working in directory: $DIR ====="


# if there is a main.py file, run it as a Streamlit app
if [ -f "main.py" ]; then
    echo "Found \"main.py\" -- starting the Streamlit app."
    streamlit_config=$(streamlit config show 2>/dev/null)
    port=$(echo "$streamlit_config" | grep "port = " | grep -o '[0-9]\+' | sed 's/^0*//')
    baseUrlPath=$(echo "$streamlit_config" | grep "baseUrlPath = " | sed 's/baseUrlPath = "//;s/"$//')
    serverAddress=$(echo "$streamlit_config" | grep "serverAddress = " | sed 's/serverAddress = "//;s/"$//')

    if [ -n "$port" ]; then
        if [ -n "$(lsof -i :$port)" ]; then
            if [[ "$FORCE" = true || "$KILL_SERVICES" = true ]]; then
                echo "  Port $port is already in use -- shutting down the running Streamlit app."
                kill -9 $(lsof -t -i:$port)
            else
                echo "  Port $port is already in use -- skipping the Streamlit app."
                runStreamlit=false
            fi
        fi
    fi
else
    echo "No \"main.py\" file found -- skipping the Streamlit app."
    runStreamlit=false
fi

if ! [[ "$runStreamlit" = false || "$KILL_SERVICES" = true ]]; then
    streamlit run main.py --browser.serverAddress "$SERVER_ADDRESS" > streamlit.log 2>&1 &
    echo "  Hosting the Streamlit app in localhost:$port/$baseUrlPath"
    echo "  If you have a reverse proxy (e.g., Nginx) set up, you can access it at $SERVER_ADDRESS/$baseUrlPath"
fi


# if there is a Flask.py file, run it as a Flask app
if [ -f "Flask.py" ]; then
    echo "Found \"Flask.py\" -- starting the Flask app."

    flaskPort=$(grep "app.run(" Flask.py | grep -o 'port=[0-9]\+' | grep -o '[0-9]\+' | sed 's/^0*//' )
    if [ -n "$flaskPort" ]; then
        if [ -n "$(lsof -i :$flaskPort)" ]; then
            if [[ "$FORCE" = true || "$KILL_SERVICES" = true ]]; then
                echo "  Port $flaskPort is already in use -- shutting down the running Flask app."
                kill -9 $(lsof -t -i:$flaskPort)
            else
                echo "  Port $flaskPort is already in use -- skipping the Flask app."
                runFlask=false
            fi
        fi
    fi
else
    echo "No \"Flask.py\" file found -- skipping the Flask app."
    runFlask=false
fi

if ! [[ "$runFlask" = false || "$KILL_SERVICES" = true ]]; then
    python Flask.py >> flask.log 2>&1 &
    echo "  Hosting the Flask app in localhost:$flaskPort"
fi

echo "===== End of Service Logs ====="
