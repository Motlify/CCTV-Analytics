import threading


class ROITimeoutScheduler:
    def __init__(self, region_name, timeout, callback_start, callback_end):
        """
        Initializes the TimeoutScheduler.

        :param timeout: Timeout duration in seconds.
        :param callback: Function to call when the timeout is reached.
        """
        self.region_name = region_name
        self.timeout = timeout
        self.callback_start = callback_start
        self.callback_end = callback_end
        self.timer = None

    def _run_callback_end(self):
        """
        Internal method to run the end callback.
        """
        self.callback_end()
        self.timer = None

    def _run_callback_start(self):
        """
        Internal method to run the start callback.
        """
        self.callback_start()

    def start(self):
        """
        Starts or restarts the timeout.
        If a previous timeout is still running, it will be canceled.
        """
        if self.timer is None:
            self._run_callback_start()
        self.cancel()  # Cancel any existing timer
        self.timer = threading.Timer(self.timeout, self._run_callback_end)
        self.timer.start()

    def cancel(self):
        """
        Cancels the currently running timeout, if any.
        """
        if self.timer is not None:
            self.timer.cancel()
            self.timer = None
