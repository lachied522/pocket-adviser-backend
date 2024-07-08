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