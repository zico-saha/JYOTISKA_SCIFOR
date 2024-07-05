# Python Encapsulation Documentation

## Table of Contents
- [Encapsulation Basics (in Python)](#encapsulation-basics-in-python)
- [Real-Life Example of Encapsulation](#real-life-example-of-encapsulation)
- [Implementation of Encapsulation in Python](#implementation-of-encapsulation-in-python)

### 1. Encapsulation Basics (in Python)
Encapsulation is one of the fundamental concepts in object-oriented programming (OOP). The wrapping up of data and functions into a single unit (called class) is known as Encapsulation.

The data is not accessible to the outside world, and only those functions which are wrapped in the class can access it. These functions provide the interface between the object's data and the program. Inhibiting access to the data by the program is called data hiding. For interacting with the object's data, it provides a controlled and secure interface.

By hiding implementation details, encapsulation makes it easier to modify and maintain the code.

In Python, Encapsulation can be achieved using private and protected access specifiers:

- **Private Members:** The private data variable name starts with double underscore (\_\_). These members are not accessible from outside the class (not even derived classes).
- **Protected Members:** The protected data variable name starts with a single underscore (\_). These members are accessible within the class and its derived classes. Although they are accessible outside of the class too, the convention is not to access them.
- **Public Members:** The public data variables are not preceded with any underscore. Though the convention is to keep data private for security and integrity reasons, some data can be made public if it doesn't harm the program's integrity.

All data variables are initialized in two ways in Python:

- Inside the `__init__` method: Variables declared inside this method are called instance variables and are accessed using `self`.
- Inside the class and outside `__init__` method: These variables are class variables and are accessed using `cls`.

### 2. Real-Life Example of Encapsulation
Consider a scenario of an ATM. When using an ATM to withdraw money, several internal processes occur that are hidden from us. Encapsulation ensures that we interact with the ATM in a simple and secure manner without needing to know the intricate details of how it works internally.

In an ATM, we swipe the ATM card and the machine takes user information like "card number", "card holder name", "expiration date", "CVV number", etc. The user also has to provide a 4-digit pin. The user cannot access the data directly or know the internal process details. They can only work with the options (methods) provided on the ATM machine screen.

Hence, Encapsulation ensures that we interact with the ATM in a simple and secure manner without needing to know the intricate details of how it works internally.

### 3. Implementation of Encapsulation in Python

Below is an example describing encapsulation in Python:

The class `ATM_Machine` encapsulates several private static variables and methods:

- **Private Static Variables:**
  - `account_number`
  - `balance`
  - `pin`

- **Private Methods:**
  - `verify_pin`: Deals with verifying the ATM pin.
  - `get_balance`: Retrieves the balance of the account.
  - `withdraw`: Handles the process of withdrawing money.
  - `change_pin`: Manages the process of changing the ATM pin.

The only method accessible outside the class is `options`. This encapsulation ensures that other data and methods within the class are hidden and not directly accessible to external programs.


```python
class ATM_Machine:
    def __init__(self, account_number, balance, pin):
        self.__account_number = account_number
        self.__balance = balance
        self.__pin = pin

    def __verify_pin(self, entered_pin):
        return self.__pin == entered_pin

    def __get_balance(self, entered_pin):
        if self.__verify_pin(entered_pin):
            print(f"\nCurrent Balance: {self.__balance}")
        else:
            print("\nInvalid PIN!")

    def __withdraw(self, entered_pin):
        if self.__verify_pin(entered_pin):
            print(f"\nCurrent Balance: {self.__balance}")
            
            try:
                amount = int(input("Enter the amount: "))
            except ValueError:
                print("\nError: You must enter a valid number.")
            else:
                if amount <= self.__balance:
                    self.__balance -= amount
                    print(f"\nWithdrawal successful! \nCurrent Balance: {self.__balance}")
                else:
                    print("\nInsufficient funds!")
        else:
            print("\nInvalid PIN!")

    def __change_pin(self, entered_pin):
        if self.__verify_pin(entered_pin):
            while(True):
                new_pin = input("Enter new PIN: ")

                if new_pin.isdigit() and len(new_pin) == 4:
                    self.__pin = new_pin
                    break
                else:
                    print("Invalid PIN number. Try again!")
        else:
            print("\nInvalid PIN!")
    
    def options(self):
        while(True):
            print("\n----- ATM Menu -----")
            print("1. Check Balance")
            print("2. Withdraw Cash")
            print("3. Change PIN")
            print("4. Exit")
            
            option = input("Please select an option: ")

            match(option):
                case '1':
                    pin = input("Enter your PIN: ")
                    self.__get_balance(pin)
                case '2':
                    pin = input("Enter your PIN: ")
                    self.__withdraw(pin)
                case '3':
                    pin = input("Enter your PIN: ")
                    self.__change_pin(pin)
                case '4':
                    print("\nThank you for using the ATM.")
                    return
                case _:
                    print("\nInvalid option selected.")



user = ATM_Machine("12345678", 5000, "2468")
user.options()
```

### OUTPUT:

----- ATM Menu -----
1. Check Balance
2. Withdraw Cash
3. Change PIN
4. Exit
Please select an option: 1
Enter your PIN: 2468

Current Balance: 5000

----- ATM Menu -----
1. Check Balance
2. Withdraw Cash
3. Change PIN
4. Exit
Please select an option: 1
Enter your PIN: 2274

Invalid PIN!

----- ATM Menu -----
1. Check Balance
2. Withdraw Cash
3. Change PIN
4. Exit
Please select an option: 2
Enter your PIN: 2468

Current Balance: 5000
Enter the amount: 500

Withdrawal successful! 
Current Balance: 4500

----- ATM Menu -----
1. Check Balance
2. Withdraw Cash
3. Change PIN
4. Exit
Please select an option: 2
Enter your PIN: 2468

Current Balance: 4500
Enter the amount: 10000

Insufficient funds!

----- ATM Menu -----
1. Check Balance
2. Withdraw Cash
3. Change PIN
4. Exit
Please select an option: 3
Enter your PIN: 2561

Invalid PIN!

----- ATM Menu -----
1. Check Balance
2. Withdraw Cash
3. Change PIN
4. Exit
Please select an option: 3
Enter your PIN: 2468
Enter new PIN: 1234

----- ATM Menu -----
1. Check Balance
2. Withdraw Cash
3. Change PIN
4. Exit
Please select an option: 1
Enter your PIN: 2468

Invalid PIN!

----- ATM Menu -----
1. Check Balance
2. Withdraw Cash
3. Change PIN
4. Exit
Please select an option: 1
Enter your PIN: 1234

Current Balance: 4500

----- ATM Menu -----
1. Check Balance
2. Withdraw Cash
3. Change PIN
4. Exit
Please select an option: 5
Invalid option selected.

----- ATM Menu -----
1. Check Balance
2. Withdraw Cash
3. Change PIN
4. Exit
Please select an option: 4
Thank you for using the ATM.