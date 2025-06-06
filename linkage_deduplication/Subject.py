'''
Author: Michael Kurilko
Date: 6/2/2025
Description: This module contains the Subject class, which represents a subject in the linkage and deduplication process.
This class is used to store and manage the attributes of a subject, including their name, date of birth, and other relevant information.
'''

class Subject:
    def __init__(self, first_name, last_name, dob, email, phone_number=None, dod=None, middle_name=None, birth_city=None, attributes=None):
        """
        Initialize a Subject instance.

        :param first_name: The first name of the subject.
        :param middle_name: The middle name of the subject.
        :param last_name: The last name of the subject.
        :param dob: The date of birth of the subject.
        :param dod: The date of death of the subject (optional).
        :param email: The email address of the subject.
        :param phone_number: The phone number of the subject (optional).
        :param birth_city: The city of birth of the subject.
        :param attributes: A dictionary of additional attributes for the subject.
        """
        self.first_name = first_name
        self.middle_name = middle_name
        self.last_name = last_name
        self.dob = dob
        self.dod = dod
        self.email = email
        self.phone_number = phone_number
        self.birth_city = birth_city
        # Attributes can include any additional information relevant to the subject
        self.attributes = attributes if attributes is not None else {}

    def __repr__(self):
        """
        Return a string representation of the Subject instance.
        """
        return (f"Subject(first_name={self.first_name}, middle_name={self.middle_name}, "
                f"last_name={self.last_name}, dob={self.dob}, dod={self.dod}, email={self.email}, "
                f"phone_number={self.phone_number}, birth_city={self.birth_city}, attributes={self.attributes})")