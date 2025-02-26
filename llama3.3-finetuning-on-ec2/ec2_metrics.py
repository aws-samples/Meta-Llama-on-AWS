"""
Collects and logs GPU and CPU metrics to an EC2 metrics CSV file. This file is populated during the
duration of the inferences against the model deployed on the EC2 instance.
"""
import os
import csv
import time
import nvitop
import psutil
import logging
import globals as g
from transformers import TrainerCallback
from nvitop import Device, ResourceMetricCollector

# Setup logging
logging.basicConfig(
    format="[%(asctime)s] p%(process)s {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Update the metrics file path to be in the results directory
METRICS_FILE_PATH = os.path.join(g.RESULTS_DIR, g.EC2_SYSTEM_METRICS_FNAME)

# Global flag to control data collection
collecting = True

def stop_collect(collector=None):
    """
    Stops the data collection process by setting the global flag 'collecting' to False.
    """
    global collecting
    collecting = False
    logger.info("Stopped collection")


def _collect_ec2_utilization_metrics():
    """
    Starts the data collection process by initializing the ResourceMetricCollector and collecting metrics at regular intervals.
    """
    global collecting
    logger.info("Starting collection")

    def on_collect(metrics):
        """
        Collects GPU and CPU metrics, then appends them to the CSV file.
        Returns False if the collection should stop.
        """
        if not collecting:
            return False

        try:
            # Open the CSV file in append mode and write the collected metrics
            with open(METRICS_FILE_PATH, mode="a", newline="") as csv_file:
                csv_writer = csv.writer(csv_file)

                # Collect CPU mean utilization
                cpu_percent_mean = metrics.get(
                    "metrics-daemon/host/cpu_percent (%)/mean", psutil.cpu_percent()
                )
                memory_percent_mean = metrics.get(
                    "metrics-daemon/host/memory_percent (%)/mean",
                    psutil.virtual_memory().percent,
                )
                memory_used_mean = metrics.get(
                    "metrics-daemon/host/memory_used (GiB)/mean",
                    psutil.virtual_memory().available,
                )

                # Extract the current timestamp
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

                # Initialize variables to sum GPU metrics
                total_gpu_utilization = 0
                total_gpu_memory_used = 0
                total_gpu_memory_free = 0
                total_gpu_memory_total = 0

                num_gpus = len(Device.cuda.all())

                # Iterate over all detected GPUs
                for gpu_id in range(num_gpus):
                    gpu_utilization_mean = metrics.get(
                        f"metrics-daemon/cuda:{gpu_id} (gpu:{gpu_id})/gpu_utilization (%)/mean",
                        None,
                    )
                    gpu_memory_used_mean = metrics.get(
                        f"metrics-daemon/cuda:{gpu_id} (gpu:{gpu_id})/memory_used (MiB)/mean",
                        None,
                    )
                    gpu_memory_free_mean = metrics.get(
                        f"metrics-daemon/cuda:{gpu_id} (gpu:{gpu_id})/memory_free (MiB)/mean",
                        None,
                    )
                    gpu_memory_total_mean = metrics.get(
                        f"metrics-daemon/cuda:{gpu_id} (gpu:{gpu_id})/memory_total (MiB)/mean",
                        None,
                    )

                    if gpu_utilization_mean is not None:
                        total_gpu_utilization += gpu_utilization_mean
                    if gpu_memory_used_mean is not None:
                        total_gpu_memory_used += gpu_memory_used_mean
                    if gpu_memory_free_mean is not None:
                        total_gpu_memory_free += gpu_memory_free_mean
                    if gpu_memory_total_mean is not None:
                        total_gpu_memory_total += gpu_memory_total_mean

                # Calculate the mean values across all GPUs (if available)
                gpu_utilization_mean_total = (
                    total_gpu_utilization / num_gpus if num_gpus > 0 else None
                )
                gpu_memory_used_mean_total = (
                    total_gpu_memory_used / num_gpus if num_gpus > 0 else None
                )
                gpu_memory_free_mean_total = (
                    total_gpu_memory_free / num_gpus if num_gpus > 0 else None
                )
                gpu_memory_total_mean_total = (
                    total_gpu_memory_total / num_gpus if num_gpus > 0 else None
                )

                # Write the row to the CSV file
                row = [
                    timestamp,
                    cpu_percent_mean,
                    memory_percent_mean,
                    memory_used_mean,
                    gpu_utilization_mean_total,
                    gpu_memory_used_mean_total,
                    gpu_memory_free_mean_total,
                    gpu_memory_total_mean_total,
                ]
                csv_writer.writerow(row)

        except ValueError as e:
            logger.error(f"Error writing metrics: {e}")
            return False

        return True

    # Start the collector and run in the background
    collector = ResourceMetricCollector(Device.cuda.all())
    logger.info("Starting daemon collector to run in background")
    collector.daemonize(
        on_collect,
        interval=g.EC2_UTILIZATION_METRICS_INTERVAL,
        on_stop=stop_collect,
    )


def collect_ec2_metrics():
    """
    Initializes the CSV file with headers and starts the metrics collection process.
    """
    global collecting
    collecting = True
    # Initialize the CSV file and write the header once
    with open(METRICS_FILE_PATH, mode="w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        header = [
            "timestamp",
            "cpu_percent_mean",
            "memory_percent_mean",
            "memory_used_mean",
            "gpu_utilization_mean",
            "gpu_memory_used_mean",
            "gpu_memory_free_mean",
            "gpu_memory_total_mean",
        ]
        logger.info(f"Writing header: {header}")
        csv_writer.writerow(header)

    # Start collecting metrics in the background
    _collect_ec2_utilization_metrics()

class EC2MetricsCallback(TrainerCallback):
    """
    A custom TrainerCallback that starts EC2 metrics collection at the beginning
    of training and stops it at the end. This callback class is used in the trainer
    to call the on train begin and on train end to get the GPU/CPU utilization metrics
    """

    def on_train_begin(self, args, state, control, **kwargs):
        logger.info("Training started. Initiating EC2 metrics collection.")
        # Ensure results directory exists before starting collection
        os.makedirs(RESULTS_DIR, exist_ok=True)
        collect_ec2_metrics()
        return control

    def on_train_end(self, args, state, control, **kwargs):
        logger.info("Training ended. Stopping EC2 metrics collection.")
        stop_collect()
        return control
