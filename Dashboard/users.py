from flask_login import UserMixin

class User(UserMixin):
    def __init__(self, id, username, password, is_admin=False):
        self.id = id
        self.username = username
        self.password = password
        self.is_admin = is_admin

# Example users (replace with DB integration as needed)
users = {
    'user1': User(1, 'user1', 'userpass', False),
    'admin': User(2, 'admin', 'adminpass', True)
}

def get_user(username):
    return users.get(username)
