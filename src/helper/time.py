from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo, available_timezones


def get_time(pattern="[DAY.MONTH.YEAR, HOUR:MINUTE]",
               offset_days=0,
               offset_hours=0,
               offset_minutes=0,
               offset_seconds=0,
               time_zone="Europe/Berlin") -> str:
    """
    Prints the current time with a given offset and with a given pattern.

    ---
    Parameters:
    - pattern : str, optional (default='[DAY.MONTH.YEAR, HOUR:MINUTE]')
        Use the given pattern to print the date/time.
        Use the keywords DAY, MONTH, YEAR, HOUR, MINUTE, SECOND where the date/time value should be placed.
    - offset_days : int, optional (default=0)
        Defines the day offset to the given current time.
    - offset_hours : int, optional (default=0)
        Defines the hour offset to the given current time.
    - offset_minutes : int, optional (default=0)
        Defines the minute offset to the given current time.
    - offset_seconds : int, optional (default=0)
        Defines the second offset to the given current time.
    - timezone : Union(str, None), optional (default="Europe/Berlin")
        Defines the timezone after pythons internal datetime lib. None for no timezone usage -> UTC + 0.
    """
    if time_zone and time_zone in available_timezones():
        now = datetime.now(ZoneInfo("Europe/Berlin"))
    else:
        now = datetime.now()

    now_with_offset = now + timedelta(days=offset_days, hours=offset_hours, minutes=offset_minutes, seconds=offset_seconds)
    print_str = pattern.replace("DAY", f"{now_with_offset.day:02}")\
                       .replace("MONTH", f"{now_with_offset.month:02}")\
                       .replace("YEAR", f"{now_with_offset.year:04}")\
                       .replace("HOUR", f"{now_with_offset.hour:02}")\
                       .replace("MINUTE", f"{now_with_offset.minute:02}")\
                       .replace("SECOND", f"{now_with_offset.second:02}")
    return print_str



