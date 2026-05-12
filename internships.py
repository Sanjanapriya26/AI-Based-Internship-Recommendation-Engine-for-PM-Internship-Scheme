"""
PM Internship Dataset
Contains internship listings and synthetic candidate-application data for ML training.
"""

INTERNSHIPS = [
    {"id": 1, "title": "Manufacturing Technician Trainee", "company": "Tata Motors", "icon": "🏭",
     "location_city": "Jamshedpur", "location_type": "small_city", "duration": "6 months",
     "salary": 12000, "openings": 60,
     "description": "Learn automotive manufacturing processes and quality control. Hands-on training in world-class facilities.",
     "perks": ["Certificate", "Safety gear", "Full-time offer"],
     "sectors": ["Manufacturing"], "state": "Jharkhand",
     "required_edu_min": 2,  # ITI=2
     "preferred_skills": ["Mechanical Work", "Teamwork", "Problem Solving"],
     "preferred_fields": ["Engineering", "Science"]},

    {"id": 2, "title": "Government Office Assistant", "company": "District Collectorate", "icon": "🏛️",
     "location_city": "Various Districts", "location_type": "small_city", "duration": "6 months",
     "salary": 10000, "openings": 200,
     "description": "Support administrative functions in government offices. Learn public administration.",
     "perks": ["Certificate", "Govt experience", "Flexible hours"],
     "sectors": ["Government"], "state": "Pan India",
     "required_edu_min": 1,  # 12th=1
     "preferred_skills": ["Data Entry", "Communication", "MS Office / Excel"],
     "preferred_fields": ["Arts / Humanities", "Commerce", "Management"]},

    {"id": 3, "title": "Electrician Apprentice", "company": "BHEL", "icon": "⚡",
     "location_city": "Bhopal", "location_type": "big_city", "duration": "6 months",
     "salary": 11000, "openings": 40,
     "description": "Hands-on training in electrical systems and power equipment. Government PSU experience.",
     "perks": ["Certificate", "Safety gear", "Govt PSU experience"],
     "sectors": ["Energy & Power", "Manufacturing"], "state": "Madhya Pradesh",
     "required_edu_min": 2,
     "preferred_skills": ["Electrical Work", "Teamwork", "Mechanical Work"],
     "preferred_fields": ["Engineering", "Science"]},

    {"id": 4, "title": "Agricultural Field Assistant", "company": "IARI", "icon": "🌾",
     "location_city": "Various Rural Districts", "location_type": "village", "duration": "6 months",
     "salary": 10000, "openings": 200,
     "description": "Support agricultural research projects. Learn modern farming techniques.",
     "perks": ["Accommodation", "Travel allowance", "Certificate"],
     "sectors": ["Agriculture"], "state": "Pan India",
     "required_edu_min": 0,  # 10th=0
     "preferred_skills": ["Agriculture/Farming", "Teamwork"],
     "preferred_fields": ["Agriculture", "Science"]},

    {"id": 5, "title": "Bank Customer Service Intern", "company": "State Bank of India", "icon": "🏦",
     "location_city": "Pan India", "location_type": "big_city", "duration": "3 months",
     "salary": 8000, "openings": 500,
     "description": "Learn banking operations, customer service, and financial products.",
     "perks": ["Certificate", "Banking experience", "PPO opportunity"],
     "sectors": ["Banking & Finance"], "state": "Pan India",
     "required_edu_min": 1,
     "preferred_skills": ["Communication", "Customer Service", "MS Office / Excel", "Accounting"],
     "preferred_fields": ["Commerce", "Management", "Computer Science / IT"]},

    {"id": 6, "title": "IT Support Trainee", "company": "Infosys BPM", "icon": "💻",
     "location_city": "Bengaluru / Pune", "location_type": "big_city", "duration": "6 months",
     "salary": 15000, "openings": 100,
     "description": "Technical support, troubleshooting, and IT operations training.",
     "perks": ["Certificate", "Laptop provided", "Full-time offer"],
     "sectors": ["IT & Technology"], "state": "Karnataka",
     "required_edu_min": 3,  # Graduate=3
     "preferred_skills": ["Computer Basics", "Programming", "MS Office / Excel", "Problem Solving"],
     "preferred_fields": ["Computer Science / IT", "Engineering", "Science"]},

    {"id": 7, "title": "Healthcare Assistant", "company": "Apollo Hospitals", "icon": "🏥",
     "location_city": "Major Cities", "location_type": "big_city", "duration": "6 months",
     "salary": 9000, "openings": 80,
     "description": "Assist medical staff in patient care and hospital administration.",
     "perks": ["Certificate", "Meals provided", "Employment opportunity"],
     "sectors": ["Healthcare"], "state": "Pan India",
     "required_edu_min": 0,
     "preferred_skills": ["Healthcare", "Communication", "Teamwork", "Customer Service"],
     "preferred_fields": ["Medical / Nursing", "Science"]},

    {"id": 8, "title": "Retail Sales Associate", "company": "Reliance Retail", "icon": "🛒",
     "location_city": "Pan India", "location_type": "small_city", "duration": "3 months",
     "salary": 9500, "openings": 300,
     "description": "Learn retail operations, customer handling, and inventory management.",
     "perks": ["Certificate", "Staff discount", "Incentive bonus"],
     "sectors": ["Retail & Sales"], "state": "Pan India",
     "required_edu_min": 0,
     "preferred_skills": ["Customer Service", "Communication", "Teamwork"],
     "preferred_fields": ["Commerce", "Arts / Humanities", "Management"]},

    {"id": 9, "title": "Data Entry Operator", "company": "NIC (Govt of India)", "icon": "🖥️",
     "location_city": "District Offices", "location_type": "small_city", "duration": "6 months",
     "salary": 10000, "openings": 150,
     "description": "Enter and manage government data on national databases. Learn e-governance.",
     "perks": ["Certificate", "Govt experience", "Flexible hours"],
     "sectors": ["Government", "IT & Technology"], "state": "Pan India",
     "required_edu_min": 0,
     "preferred_skills": ["Data Entry", "Computer Basics", "MS Office / Excel"],
     "preferred_fields": ["Computer Science / IT", "Commerce", "Arts / Humanities"]},

    {"id": 10, "title": "Logistics & Supply Chain Intern", "company": "Amazon India", "icon": "🚚",
     "location_city": "Fulfillment Centers", "location_type": "big_city", "duration": "3 months",
     "salary": 13000, "openings": 250,
     "description": "Warehouse operations, inventory tracking, and supply chain management.",
     "perks": ["Certificate", "Transport facility", "Pre-placement offer"],
     "sectors": ["Logistics"], "state": "Pan India",
     "required_edu_min": 1,
     "preferred_skills": ["Teamwork", "Computer Basics", "Problem Solving", "MS Office / Excel"],
     "preferred_fields": ["Engineering", "Commerce", "Management"]},

    {"id": 11, "title": "Solar Panel Technician Trainee", "company": "Adani Green Energy", "icon": "☀️",
     "location_city": "Rajasthan / Gujarat", "location_type": "small_city", "duration": "6 months",
     "salary": 11500, "openings": 75,
     "description": "Installation and maintenance of solar panels. Renewable energy sector experience.",
     "perks": ["Certificate", "Safety gear", "Full-time offer"],
     "sectors": ["Energy & Power", "Manufacturing"], "state": "Rajasthan",
     "required_edu_min": 2,
     "preferred_skills": ["Electrical Work", "Mechanical Work", "Teamwork"],
     "preferred_fields": ["Engineering", "Science"]},

    {"id": 12, "title": "Rural Banking Correspondent", "company": "NABARD", "icon": "🏧",
     "location_city": "Rural Areas", "location_type": "village", "duration": "6 months",
     "salary": 9000, "openings": 400,
     "description": "Provide banking services in rural communities. Financial inclusion mission.",
     "perks": ["Certificate", "Travel allowance", "Govt experience"],
     "sectors": ["Banking & Finance", "Government"], "state": "Pan India",
     "required_edu_min": 1,
     "preferred_skills": ["Communication", "Accounting", "Customer Service"],
     "preferred_fields": ["Commerce", "Agriculture", "Arts / Humanities"]},

    {"id": 13, "title": "Teacher's Assistant (Govt Schools)", "company": "Ministry of Education", "icon": "📚",
     "location_city": "Various Districts", "location_type": "small_city", "duration": "6 months",
     "salary": 8500, "openings": 600,
     "description": "Assist teachers in government schools. Help students with digital literacy.",
     "perks": ["Certificate", "Govt experience", "Community impact"],
     "sectors": ["Education", "Government"], "state": "Pan India",
     "required_edu_min": 1,
     "preferred_skills": ["Teaching", "Communication", "Computer Basics"],
     "preferred_fields": ["Arts / Humanities", "Science", "Computer Science / IT"]},

    {"id": 14, "title": "Hospitality Trainee", "company": "Taj Hotels", "icon": "🏨",
     "location_city": "Major Tourist Cities", "location_type": "big_city", "duration": "6 months",
     "salary": 10000, "openings": 120,
     "description": "Front office, housekeeping, and F&B operations in luxury hospitality.",
     "perks": ["Certificate", "Meals provided", "Grooming allowance"],
     "sectors": ["Hospitality", "Retail & Sales"], "state": "Pan India",
     "required_edu_min": 1,
     "preferred_skills": ["Customer Service", "Communication", "Teamwork"],
     "preferred_fields": ["Arts / Humanities", "Management", "Commerce"]},

    {"id": 15, "title": "Construction Site Assistant", "company": "L&T Construction", "icon": "🏗️",
     "location_city": "Major Projects", "location_type": "big_city", "duration": "6 months",
     "salary": 12000, "openings": 90,
     "description": "Site management support, safety monitoring, and construction project tracking.",
     "perks": ["Certificate", "Safety gear", "PPF contribution"],
     "sectors": ["Construction", "Manufacturing"], "state": "Pan India",
     "required_edu_min": 2,
     "preferred_skills": ["Mechanical Work", "Teamwork", "Problem Solving"],
     "preferred_fields": ["Engineering", "Science"]},
]

# Education encoding map
EDU_MAP = {
    "10th": 0, "12th": 1, "ITI": 2, "Diploma": 2,
    "Graduate": 3, "PostGraduate": 4
}

# All possible features
ALL_SKILLS = [
    "Computer Basics", "MS Office / Excel", "Programming", "Data Entry",
    "Mechanical Work", "Electrical Work", "Communication", "Teamwork",
    "Customer Service", "Leadership", "Problem Solving", "Driving License",
    "Accounting", "Teaching", "Healthcare", "Agriculture/Farming"
]

ALL_SECTORS = [
    "IT & Technology", "Manufacturing", "Healthcare", "Agriculture",
    "Banking & Finance", "Education", "Retail & Sales", "Construction",
    "Hospitality", "Government", "Logistics", "Energy & Power"
]

ALL_FIELDS = [
    "Arts / Humanities", "Science", "Commerce", "Engineering",
    "Medical / Nursing", "Agriculture", "Computer Science / IT",
    "Management", "Law", "Other"
]

LOCATION_TYPES = ["big_city", "small_city", "village", "work_from_home", "anywhere"]
