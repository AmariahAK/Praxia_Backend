from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import structlog

logger = structlog.get_logger(__name__)

def run_with_timeout(func, timeout_seconds=30, *args, **kwargs):
    """
    Run a function with a timeout using ThreadPoolExecutor
    
    Args:
        func: Function to run
        timeout_seconds: Timeout in seconds
        *args, **kwargs: Arguments to pass to the function
    
    Returns:
        Function result or raises TimeoutError
    """
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout_seconds)
        except FutureTimeoutError:
            future.cancel()
            logger.warning("Function execution timed out", 
                         function=func.__name__, 
                         timeout=timeout_seconds)
            raise TimeoutError(f"Function {func.__name__} timed out after {timeout_seconds} seconds")
