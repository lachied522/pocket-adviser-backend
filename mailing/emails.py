from mailersend import emails

mailer = emails.NewEmail() # api key is fetched implicitly

async def send_email(
    recipient_name: str,
    recipient_email: str,
    subject: str,
    content: str,
    sender_name = "Pocket Adviser",
    sender_email = "pocketadviser@pocketadviser.com.au"
):
    # see https://github.com/mailersend/mailersend-python?tab=readme-ov-file#authentication
    # define an empty dict to populate with mail values
    try:
        mail_body = {}

        mail_from = {
            "name": sender_name,
            "email": sender_email,
        }

        recipients = [
            {
                "name": recipient_name,
                "email": recipient_email,
            }
        ]

        mailer.set_mail_from(mail_from, mail_body)
        mailer.set_mail_to(recipients, mail_body)
        mailer.set_subject(subject, mail_body)
        mailer.set_html_content(content, mail_body)
        mailer.set_plaintext_content(content, mail_body) # TO DO:

        # using print() will also return status code and data
        mailer.send(mail_body)
    except Exception as e:
        print("Could not send email: ", str(e))