import msvcrt
import threading


class ExitException(Exception):
    pass


def listen_for_enter():
    while True:
        if msvcrt.kbhit():
            if msvcrt.getch() == b'\r':
                raise ExitException()


listener_thread = threading.Thread(target=listen_for_enter)
listener_thread.start()


try:
    while True:
        # Perform your loop operations here
        pass
except ExitException:
    # Handle the Enter key press and break out of the loop
    print("Enter key pressed. Exiting loop.")

listener_thread.join()  # Wait for the listener thread to finish

