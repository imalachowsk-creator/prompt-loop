# datetime

## Current date and time
datetime.now() returns the current local date and time and date.today() returns today's date. For an aware timestamp in UTC use datetime.now(timezone.utc).

## Parsing and formatting
datetime.strptime(text, "%Y-%m-%d") parses a string into a datetime using a format, and dt.strftime("%d.%m.%Y") formats a datetime as a string. datetime.fromisoformat("2026-11-05") parses ISO 8601 strings directly.

## Time differences
Subtracting two datetimes gives a timedelta. Add timedelta(days=7) to a date to move it a week forward. timedelta.total_seconds() returns the difference in seconds.

## Time zones
A naive datetime has no time zone and an aware one has tzinfo set. The zoneinfo module provides IANA zones, for example ZoneInfo("Europe/Warsaw"). Convert between zones with dt.astimezone(ZoneInfo("UTC")).
