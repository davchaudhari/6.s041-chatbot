import time
import json

from bs4 import BeautifulSoup

from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver import ActionChains
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from webdriver_manager.chrome import ChromeDriverManager

from pydantic import BaseModel, Field, ValidationError, field_validator

class UserInput(BaseModel):
    grade_level: str = Field(..., alias="grade")
    street_number: str = Field(..., alias="street_number")
    street_name: str = Field(..., alias="street_name")
    zip_code: str = Field(..., alias="zip_code")  # changed alias to "zip_code"

    @field_validator("grade_level", mode="before")
    def ensure_str(cls, v):
        # Convert numeric input to string if necessary.
        return str(v)
    
    class Config:
        allow_population_by_field_name = True

# Define the Pydantic schema.
class SchoolDetails(BaseModel):
    school_name: str = Field(..., alias="School Name")
    address: str = Field(..., alias="Address")
    school_website: str = Field("", alias="School Website")
    email: str = Field("", alias="Email")
    distance_from_home: str = Field(..., alias="Distance from home")
    school_hours: str = Field("", alias="School Hours")
    preview_dates: str = Field("", alias="Preview Dates")
    surround_care: str = Field("", alias="Surround Care")
    school_description: str = Field("", alias="School Description")
    grades_offered: str = Field("", alias="Grades Offered")
    demand_reports: str = Field("", alias="Demand Reports")
    eligibility: str = Field("", alias="Eligibility")
    special_application: str = Field("", alias="Special Application")
    quality: str = Field("", alias="Quality")
    uniform_policy: str = Field("", alias="Uniform Policy")
    school_focus: str = Field("", alias="School Focus")
    programs: str = Field("", alias="Programs")
    facility_features: str = Field("", alias="Facility Features")
    student_support: str = Field("", alias="Student Support")
    sports: str = Field("", alias="Sports")
    community_partners: str = Field("", alias="Community Partners")

    class Config:
        allow_population_by_field_name = True


class BPSScraperAgent:
    def __init__(self, student_data: UserInput):
        """
        student_data: a dictionary containing keys:
          'grade', 'street_number', 'street_name', 'zip_code'
        """
        self.student_data = student_data
        self.base_url = "http://discoverbps.bostonpublicschools.org"
        self.driver = None

    def start(self):
        """Starts the Selenium WebDriver in headless mode and opens the target URL."""
        options = webdriver.ChromeOptions()
        # Headless mode (no GUI)
        # options.binary_location = "/usr/bin/chromium-browser"
        options.add_argument("--headless")        # Chrome 109+ headless
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")

        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=options)
        self.driver.get(self.base_url)

        # Wait until the page is fully loaded.
        WebDriverWait(self.driver, 20).until(
            lambda d: d.execute_script("return document.readyState") == "complete"
        )
        print("Page is fully loaded.")

    def fill_form(self):
        """Fills in the search form using the student_data provided."""
        grade_el = self.driver.find_element(By.ID, "grade_level_placeholder")
        street_num_el = self.driver.find_element(By.ID, "street_number")
        street_el = self.driver.find_element(By.ID, "street_name")
        zipcode_el = self.driver.find_element(By.ID, "zipcode")

        # A short pause to let elements settle.
        time.sleep(0.5)
        grade_el.send_keys(str(self.student_data.grade_level))
        street_num_el.send_keys(str(self.student_data.street_number))
        street_el.send_keys(str(self.student_data.street_name))
        zipcode_el.send_keys(str(self.student_data.zip_code))

    def keep_clicking(self, button_id, target_element_id, click_interval=2, max_attempts=20) -> bool:
        """
        Repeatedly clicks on a button (by button_id) until the target element (by target_element_id)
        appears, or until max_attempts is reached.
        """
        attempts = 0
        while attempts < max_attempts:
            try:
                target = self.driver.find_element(By.ID, target_element_id)
                if target.is_displayed():
                    print(f"Target element '{target_element_id}' found after {attempts} attempts.")
                    return True
            except NoSuchElementException:
                pass

            try:
                time.sleep(1)  # Short pause before clicking.
                button = self.driver.find_element(By.ID, button_id)
                button.click()
                print(f"Clicked button '{button_id}' (attempt {attempts + 1}).")
            except NoSuchElementException:
                print(f"Button '{button_id}' not found. It might not be available on this page.")
                return False
            except Exception as e:
                print(f"Error clicking button '{button_id}' at attempt {attempts + 1}: {e}")

            time.sleep(click_interval)
            attempts += 1

        print(f"Target element '{target_element_id}' not found after {max_attempts} attempts.")
        return False

    def scrape_school_details(self, school_details):
        """
        Extracts school details information from the school_details container.
        Expected keys include:
            - Address: First line of the first <p> element.
            - School Website: href of the <a> with title "School Website".
            - Email: Email address from the <a> with title "Email Address" (without the mailto:).
            - Distance from home: From the second <p>.
            - Plus any additional title/description pairs.
        """
        details = {}
        try:
            container_elems = school_details.find_elements(By.CSS_SELECTOR, "p")
        except Exception:
            container_elems = []
        try:
            first_p = school_details.find_element(By.CSS_SELECTOR, "p")
            lines = first_p.text.splitlines()
            address = lines[0].strip() if lines and lines[0].strip() else ""
            details["Address"] = address
        except NoSuchElementException:
            details["Address"] = ""
        try:
            website_link = school_details.find_element(By.CSS_SELECTOR, 'a[title="School Website"]')
            website_href = website_link.get_attribute("href") or ""
            details["School Website"] = website_href
        except NoSuchElementException:
            details["School Website"] = ""
        try:
            email_link = first_p.find_element(By.CSS_SELECTOR, 'a[title="Email Address"]')
            email_href = email_link.get_attribute("href") or ""
            if email_href.startswith("mailto:"):
                email_href = email_href[len("mailto:"):]
            details["Email"] = email_href
        except NoSuchElementException:
            details["Email"] = ""
        if len(container_elems) > 1:
            details["Distance from home"] = container_elems[1].text.strip()
        else:
            details["Distance from home"] = ""
        current_title = None
        for elem in container_elems[2:]:
            try:
                classes = elem.get_attribute("class").split()
            except Exception:
                classes = []
            text = elem.text.strip() if elem.text else ""
            if "title" in classes and text:
                current_title = text
            elif "descrip" in classes and "light" in classes and current_title:
                details[current_title] = text
                current_title = None
        return details

    def scrape_school_info(self, school_info):
        """
        Extracts additional school information from the school_info container.
        Expected keys include:
            - School Description
            - Any attributes found in boxes (e.g., Grades Offered, Demand Reports, etc.)
        """
        info = {}
        try:
            desc_elem = school_info.find_element(By.CLASS_NAME, 'school_description')
            info["School Description"] = desc_elem.text.strip()
        except NoSuchElementException:
            info["School Description"] = ""
        try:
            attribute_lst = school_info.find_elements(By.CLASS_NAME, 'box')
            for el in attribute_lst:
                try:
                    key = el.find_element(By.CLASS_NAME, 'title').text.strip()
                except NoSuchElementException:
                    key = ""
                try:
                    value = el.find_element(By.CLASS_NAME, 'descrip').text.strip()
                except NoSuchElementException:
                    value = ""
                if key:
                    info[key] = value
            if "Community Partners" not in info:
                info["Community Partners"] = ""
        except Exception:
            pass
        return info

    def scrape_school(self, school_el):
        """
        Combines information from school_details and school_info, along with the School Name.
        Returns a dictionary corresponding to one school's data.
        """
        school_data = {}
        try:
            school_name_elem = school_el.find_element(By.CLASS_NAME, 'school_name')
            school_data["School Name"] = school_name_elem.text.strip()
        except NoSuchElementException:
            school_data["School Name"] = ""
        try:
            school_details = school_el.find_element(By.CLASS_NAME, 'school_details')
            details_dict = self.scrape_school_details(school_details)
            school_data.update(details_dict)
        except NoSuchElementException:
            school_data.setdefault("Address", "")
            school_data.setdefault("School Website", "")
            school_data.setdefault("Email", "")
            school_data.setdefault("Distance from home", "")
        try:
            school_info = school_el.find_element(By.CLASS_NAME, 'school_info')
            info_dict = self.scrape_school_info(school_info)
            school_data.update(info_dict)
        except NoSuchElementException:
            school_data.setdefault("School Description", "")
        return school_data

    def scrape_schools(self):
        """
        Finds the list of school elements on the page,
        clicks each to open its details, and scrapes the data.
        Returns a list of validated SchoolDetails instances.
        """
        time.sleep(3)
        home_schools_lst = self.driver.find_elements(
            By.CSS_SELECTOR, "ul#sortable > li.list_row.home_school.clearfix"
        )
        time.sleep(3)
        school_data_lst = []
        for el in home_schools_lst:
            try:
                # Click the school name element to open details.
                el.find_element(By.CLASS_NAME, 'sortable_school_name').click()
                time.sleep(0.3)
                scraped_info = self.scrape_school(el)
                try:
                    # Validate using the Pydantic model.
                    school = SchoolDetails.model_validate(scraped_info)
                except ValidationError as e:
                    print("Validation error:", e)
                    continue
                print('Success for', school.school_name)
                school_data_lst.append(school)
            except Exception as e:
                print("Error processing a school element:", e)
        return school_data_lst

    def run(self):
        """
        Main method to run the scraper:
         - Starts the driver and loads the page.
         - Fills out the search form.
         - Navigates through multi-step process by repeatedly clicking buttons.
         - Scrapes and returns a list of school details.
        """
        self.start()
        # Ensure the page is fully loaded.
        WebDriverWait(self.driver, 20).until(
            lambda d: d.execute_script("return document.readyState") == "complete"
        )
        print("Page is fully loaded.")
        self.fill_form()
        # Navigate through the form steps.
        self.keep_clicking("home_search_button", "addresses_next_button")
        time.sleep(1)
        self.keep_clicking("addresses_next_button", "ell_next_button")
        time.sleep(1)
        self.keep_clicking("ell_next_button", "sped_next_button")
        time.sleep(1)
        self.keep_clicking("sped_next_button", "home_schools_list")
        time.sleep(1)
        # Scrape the school list.
        schools = self.scrape_schools()
        return schools

    def close(self):
        if self.driver:
            self.driver.quit()