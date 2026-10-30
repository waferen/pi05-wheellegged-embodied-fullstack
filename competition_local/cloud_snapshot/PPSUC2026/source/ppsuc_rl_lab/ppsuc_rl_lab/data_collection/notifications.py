"""Non-blocking Kit notifications; failures never interrupt robot control."""
import logging


def create_notifier():
    try:
        from isaaclab.utils.extensions import enable_extension
        enable_extension("omni.kit.notification_manager")
        import omni.kit.notification_manager as nm
    except Exception:
        logging.exception("Collection UI notifications unavailable; terminal messages remain active")
        return lambda message, error=False: None

    def notify(message, error=False):
        try:
            nm.post_notification(message, duration=8 if error else 5,
                                 status=nm.NotificationStatus.WARNING if error else nm.NotificationStatus.INFO)
        except Exception:
            logging.exception("Failed to display collection notification")
    return notify
