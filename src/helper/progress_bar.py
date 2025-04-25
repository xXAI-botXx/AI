

def get_progress_bar(total, progress, should_clear=False, 
                       left_bar_char="|", right_bar_char="|", progress_char="#", empty_char=" ",
                       front_message="", back_message="",
                       size=100) -> str:
    """
    Prints one step of a progress as a progress bar.

    ---
    Parameters:
    - total : union(int, float)
        Number of the total amount of the progress (the goal number).
    - progress : union(int, float)
        Number of the current state of the progress (how far the progress).
    - should_clear : bool, optional (default=False)
        Should the console output be cleared?
    - left_bar_char : str, optional (default='|')
        Left sign of the progress bar.
    - right_bar_char : str, optional (default='|')
        Right sign of the progress bar.
    - progress_char : str, optional (default='#')
        Sign of the progress in the progress bar.
    - empty_char : str, optional (default=' ')
        Sign of the missing progress in the progress bar.
    - front_message : str, optional (default="")
        Message for the progress bar.
    - back_message : str, optional (default="")
        Message behind the progress bar.
    - size : int, optional (default=100)
        Amount of signs of the progress bar.
    - should_print : bool, optional (default=True)
        Whether to print the progress bar.

    ---
    Returns:
    - str
        Created progress bar.
    """
    # clearing
    if should_clear:
        clear()
    
    # calc progress bar
    percentage = progress/float(total)
    percentage_adjusted = int( size * percentage )
    bar = progress_char*percentage_adjusted + empty_char*(size-percentage_adjusted)
    progress_str = f"{front_message} {left_bar_char}{bar}{right_bar_char} {percentage*100:0.2f}% {back_message}".strip()

    return progress_str





