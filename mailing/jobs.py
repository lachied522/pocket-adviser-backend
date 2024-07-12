import os
import traceback
from datetime import datetime

from database import SessionLocal
from models import User

from mailing.content import construct_html_body_for_email
from mailing.email import send_bulk_email

DIRECTORY = "temp"

def get_users_by_email_frequency(freq: str):
    db = SessionLocal()
    users = db.query(User).filter(User.mailFrequency == freq).all()
    # close db and return
    db.close()
    return users

def send_queued_emails(users: list[User], subject: str):
    # emails are sent in bulk to reduce API usage
    mail_list = []
    to_remove = [] # array of files to remove
    for user in users:
        # check if file exists
        file_name = f"{DIRECTORY}/{user.id}.html"
        if not os.path.isfile(file_name):
            print("File does not exist for user, ", user.id)
            continue
        # open file
        with open(file_name, 'r', encoding='utf-8') as file:
            html_content = file.read()
        
        # append to mailing list
        recipient = {
            "email": user.email
        }
        if user.name:
            recipient["name"] = user.name
        # append to mail_list
        mail_list.append({
            "to": [recipient],
            "subject": subject,
            "html": html_content,
        })
        # append file to remove array
        to_remove.append(file_name)

    send_bulk_email(mail_list)

    # cleanup files
    for file_name in to_remove:
        os.remove(file_name)

async def send_emails_by_frequency(frequencies: str|list[str]):
    if isinstance(frequencies, str):
        frequencies = [frequencies]

    # Create the directory if it doesn't exist
    os.makedirs(DIRECTORY, exist_ok=True)
    # initiliase email subject - same for each user
    subject = "Market Update {}".format(datetime.now().strftime('%#d %B %Y'))

    for freq in frequencies:
        try:
            print("Sending {} emails".format(freq.lower()))
            # get users
            users = get_users_by_email_frequency(freq)
            # queue emails, send when length is 10
            queue = []
            for user in users:
                try:
                    if not user.email:
                        # this shouldn't happen since all paid users require an email
                        raise Exception("User missing email")

                    # add file to html content to temp folder
                    file_path = f"{DIRECTORY}/{user.id}.html"
                    await construct_html_body_for_email(user, file_path=file_path)
                    # append user to email queue
                    queue.append(user)
                    # check if length queue > 10
                    if len(queue) > 10:
                        # send queued emails
                        send_queued_emails(queue, subject)
                        # reset queue
                        queue = []

                except Exception as e:
                    print(f"Could not construct email for {user.id}: ", str(e))

            # send any remaining emails
            send_queued_emails(queue, subject)
            print("Finished sending {} emails".format(freq.lower()))
        except Exception as e:
            traceback.print_exc()