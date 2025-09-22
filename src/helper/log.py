
import os


def log(content:str, path="./logs", file_name="output.log", 
        clear_log=False, add_newline=True) -> None:
    """
    Saves your content into a log file.

    ---
    Parameters:
    - content : str
        Content which should get saved.
    - path : str, optional(default="./logs")
        Path where to save the log file.
    - file_name : str, optional (default="output.log")
        Name of the log file. With or without .log, both is ok.
    - clear_log : bool, optional (default=False)
        Whether to clear the logfile. You may want to set this to True on your first logging.
    - add_newline : bool, optional (default=True)
        Decides if a backslash should be added to the end of your content.
    """
    # check/add log ending
    if not file_name.endswith(".log"):
        file_name += ".log"

    # create folder
    os.makedirs(path, exist_ok=True)
    
    path_to_file = os.path.join(path, file_name)
    
    # clear log/choose right mode
    if clear_log:
        mode = 'w'
    else:
        mode = 'a' if os.path.exists(path_to_file) else 'w'

    # adding \n
    if add_newline:
        content += "\n"

    # logging/writing
    with open(path_to_file, mode, encoding="utf-8") as f:
        f.write(content)




