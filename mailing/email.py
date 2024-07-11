from mailersend import emails

mailer = emails.NewEmail() # api key is fetched implicitly

PLAIN_TEXT_DEFAULT = (
    "If you are seeing this message, it means your email client does not support HTML emails. " +
    "Enable html to view contents."
)

def send_email(
    recipient_name: str|None,
    recipient_email: str,
    subject: str,
    html_content: str,
    plaintext_content: str = PLAIN_TEXT_DEFAULT,
    sender_name = "Pocket Adviser",
    sender_email = "newsletter@pocketadviser.com.au"
):
    # see https://github.com/mailersend/mailersend-python?tab=readme-ov-file#authentication
    try:
        # define an empty dict to populate with mail values
        mail_body = {}

        mail_from = {
            "name": sender_name,
            "email": sender_email,
        }

        recipient = {
            "email": recipient_email
        }

        if recipient_name:
            recipient["name"] = recipient_name

        mailer.set_mail_from(mail_from, mail_body)
        mailer.set_mail_to([recipient], mail_body)
        mailer.set_subject(subject, mail_body)
        mailer.set_html_content(html_content, mail_body)
        mailer.set_plaintext_content(plaintext_content, mail_body)

        mailer.send(mail_body)
    except Exception as e:
        print("Could not send email: ", str(e))

def send_bulk_email(mail_list: list[dict]):
    # see https://github.com/mailersend/mailersend-python?tab=readme-ov-file#send-bulk-email
    try:
        for mail in mail_list:
            # populate missing fields
            if "from" not in mail:
                mail["from"] = {
                    "name": "Pocket Adviser",
                    "email": "newsletter@pocketadviser.com.au",
                }
            if "text" not in mail:
                mail["text"] = PLAIN_TEXT_DEFAULT

        mailer.send_bulk(mail_list)
    except Exception as e:
        print("Could not send bulk email: ", str(e))