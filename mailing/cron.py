import traceback
from datetime import datetime

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from database import SessionLocal
from models import User

from mailing.templating import construct_html_body_for_email
from mailing.emails import send_email

def get_users_by_email_frequency(freq: str):
    db = SessionLocal()
    users = db.query(User).filter(User.mailFrequency == freq).all() # TO DO: filter by email frequency
    # close db and return
    db.close()
    return users

async def send_all_emails(freq: str):
    try:
        # get users
        users = get_users_by_email_frequency(freq)
        # define email subject - same for each user
        subject = "Market Update {}".format(datetime.now().strftime('%#d %B %Y'))
        
        for user in users:
            try:
                if not user.email:
                    # this shouldn't happen since all paid users require an email
                    raise Exception("User missing email")

                file_path = await construct_html_body_for_email(user)

                # read file
                with open(file_path, 'r', encoding='utf-8') as file:
                    html_content = file.read()

                send_email(
                    user.name,
                    user.email,
                    subject,
                    html_content,
                )

                print(f"Email sent for {user.id}")
            except Exception as e:
                print(f"Could not send email for {user.id}: ", str(e))
        
    except Exception as e:
        traceback.print_exc()

def schedule_jobs(scheduler: AsyncIOScheduler) -> None:
    scheduler.add_job(
        send_all_emails,
        args=["DAILY"],
        trigger=CronTrigger(hour=9, minute=0, day_of_week='mon-fri'),
        id="daily_emails",
        name="Send daily emails at 9am",
        max_instances=1
    )

    scheduler.add_job(
        send_all_emails,
        args=["WEEKLY"],
        trigger=CronTrigger(hour=9, minute=0, day_of_week='mon'),
        id="weekly_emails",
        name="Send weekly emails on Monday at 9am",
        max_instances=1
    )

    scheduler.add_job(
        send_all_emails,
        args=["MONTHLY"],
        trigger=CronTrigger(hour=9, minute=0, day=1),
        id="monthly_emails",
        name="Send monthly emails on last day of month at 9am",
        max_instances=1
    )