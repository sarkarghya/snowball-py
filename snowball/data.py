"""
Survey question data for product/service discovery.

This module defines a realistic survey scenario: we want to discover which
*product or service category* best matches a respondent's needs by asking
preference questions.

The data is self-contained — no external files needed.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QuestionDef:
    text: str
    answers: list[str]


@dataclass(frozen=True)
class TargetDef:
    name: str
    description: str


class SurveyData:
    """Product/service discovery survey — 20 questions, 5 answer options, 12 targets."""

    TARGETS: list[TargetDef] = [
        TargetDef("Meal-Kit Delivery", "Weekly ingredient boxes with recipes (e.g. HelloFresh)"),
        TargetDef("Streaming Entertainment", "Video/music subscription (e.g. Netflix, Spotify)"),
        TargetDef("Online Fitness Coaching", "Virtual personal training & workout plans"),
        TargetDef("Cloud Storage", "File backup & sync service (e.g. Dropbox, iCloud)"),
        TargetDef("Language Learning App", "Mobile app for learning new languages (e.g. Duolingo)"),
        TargetDef("Pet Care Subscription", "Monthly pet food, toys & vet tele-consults"),
        TargetDef("Financial Planning Tool", "Budgeting, investing & tax-prep software"),
        TargetDef("Home Cleaning Service", "Recurring professional house cleaning"),
        TargetDef("VPN / Privacy Service", "Internet privacy & security subscription"),
        TargetDef("E-Learning Platform", "Online courses & certifications (e.g. Coursera)"),
        TargetDef("Grocery Delivery", "Same-day grocery ordering & delivery"),
        TargetDef("Coworking Membership", "Flexible shared office / desk space"),
    ]

    QUESTIONS: list[QuestionDef] = [
        QuestionDef(
            "What is your top priority when choosing a new service?",
            ["Save time", "Save money", "Learn something", "Stay healthy", "Be entertained"],
        ),
        QuestionDef(
            "How often would you use this product/service?",
            ["Daily", "A few times a week", "Weekly", "Monthly", "Occasionally"],
        ),
        QuestionDef(
            "What monthly budget feels comfortable?",
            ["Under $10", "$10–25", "$25–50", "$50–100", "Over $100"],
        ),
        QuestionDef(
            "Which device do you prefer for services?",
            ["Smartphone", "Laptop/Desktop", "Tablet", "Smart TV", "No preference"],
        ),
        QuestionDef(
            "How important is data privacy to you?",
            ["Critical", "Very important", "Somewhat important", "Not very", "Don't care"],
        ),
        QuestionDef(
            "Do you live alone or with others?",
            ["Alone", "With a partner", "With family/kids", "With roommates", "Varies"],
        ),
        QuestionDef(
            "What best describes your work situation?",
            ["Office full-time", "Remote full-time", "Hybrid", "Freelance/self-employed", "Student/not working"],
        ),
        QuestionDef(
            "How do you prefer to learn new things?",
            ["Video tutorials", "Reading/articles", "Hands-on practice", "Live classes", "I don't actively learn"],
        ),
        QuestionDef(
            "How health-conscious are you?",
            ["Very — I track everything", "Fairly — I exercise regularly", "Moderate", "Not very", "Not at all"],
        ),
        QuestionDef(
            "Do you have pets?",
            ["Dog(s)", "Cat(s)", "Other pets", "Multiple types", "No pets"],
        ),
        QuestionDef(
            "What stresses you most day-to-day?",
            ["Cooking/meal planning", "Finances", "Fitness/health", "Housework", "Boredom/entertainment"],
        ),
        QuestionDef(
            "How tech-savvy are you?",
            ["Expert", "Advanced", "Intermediate", "Beginner", "Not at all"],
        ),
        QuestionDef(
            "Which area would you most like to improve?",
            ["Nutrition", "Productivity", "Skills/education", "Relaxation", "Security/privacy"],
        ),
        QuestionDef(
            "How do you feel about subscription services in general?",
            ["Love them", "Like a few", "Neutral", "Prefer one-off purchases", "Dislike them"],
        ),
        QuestionDef(
            "What kind of content do you consume most?",
            ["Movies/TV shows", "Music/podcasts", "Online courses", "News/articles", "Social media"],
        ),
        QuestionDef(
            "How often do you cook at home?",
            ["Every day", "Most days", "A few times a week", "Rarely", "Never"],
        ),
        QuestionDef(
            "How important is physical workspace to you?",
            ["Essential — I need a dedicated space", "Important", "Nice to have", "Don't care", "I work on-the-go"],
        ),
        QuestionDef(
            "What motivates you to try a new service?",
            ["Friend recommendation", "Free trial", "Online reviews", "Brand reputation", "Price/discount"],
        ),
        QuestionDef(
            "How do you handle household chores?",
            ["Do everything myself", "Split with others", "Hire help sometimes", "Hire help regularly", "Avoid them"],
        ),
        QuestionDef(
            "What is your primary goal right now?",
            ["Save more money", "Get healthier", "Advance career/skills", "Have more free time", "Feel more secure"],
        ),
    ]

    # -----------------------------------------------------------------------
    # Simulated respondent profiles — maps target_idx -> answer distribution
    # Each row = a question; value = index of the most-likely answer for that
    # target.  Used by the simulator to generate synthetic respondents.
    # -----------------------------------------------------------------------

    # fmt: off
    PROFILES: dict[int, list[int]] = {
        #                        Q0  Q1  Q2  Q3  Q4  Q5  Q6  Q7  Q8  Q9 Q10 Q11 Q12 Q13 Q14 Q15 Q16 Q17 Q18 Q19
        0:  (  # Meal-Kit Delivery
            [0,  2,  2,  0,  2,  2,  0,  2,  1,  4,  0,  2,  0,  1,  3,  2,  3,  1,  0,  3]),
        1:  (  # Streaming Entertainment
            [4,  0,  1,  3,  2,  0,  1,  0,  3,  4,  4,  2,  3,  0,  0,  3,  3,  2,  0,  3]),
        2:  (  # Online Fitness Coaching
            [3,  1,  2,  0,  2,  0,  1,  0,  0,  4,  2,  2,  0,  1,  1,  2,  3,  1,  0,  1]),
        3:  (  # Cloud Storage
            [0,  0,  0,  1,  0,  4,  1,  1,  3,  4,  1,  0,  4,  1,  3,  4,  3,  2,  0,  4]),
        4:  (  # Language Learning App
            [2,  0,  0,  0,  2,  0,  4,  2,  3,  4,  1,  2,  2,  1,  2,  4,  3,  0,  0,  2]),
        5:  (  # Pet Care Subscription
            [0,  2,  2,  0,  3,  2,  0,  4,  2,  0,  2,  2,  0,  1,  4,  2,  3,  0,  1,  3]),
        6:  (  # Financial Planning Tool
            [1,  1,  1,  1,  1,  1,  3,  1,  3,  4,  1,  1,  1,  1,  3,  4,  3,  2,  0,  0]),
        7:  (  # Home Cleaning Service
            [0,  2,  3,  4,  3,  2,  0,  4,  3,  4,  3,  3,  3,  1,  4,  1,  3,  0,  3,  3]),
        8:  (  # VPN / Privacy Service
            [0,  0,  0,  1,  0,  0,  1,  1,  3,  4,  1,  0,  4,  1,  3,  4,  3,  2,  0,  4]),
        9:  (  # E-Learning Platform
            [2,  1,  1,  1,  2,  0,  4,  0,  3,  4,  1,  2,  2,  1,  2,  4,  3,  2,  0,  2]),
        10: (  # Grocery Delivery
            [0,  1,  2,  0,  2,  2,  1,  4,  2,  4,  0,  3,  0,  1,  3,  0,  3,  1,  0,  3]),
        11: (  # Coworking Membership
            [0,  1,  3,  1,  2,  0,  3,  2,  2,  4,  1,  1,  1,  2,  3,  4,  0,  2,  0,  2]),
    }
    # fmt: on

    @classmethod
    def n_questions(cls) -> int:
        return len(cls.QUESTIONS)

    @classmethod
    def n_answers(cls) -> int:
        return len(cls.QUESTIONS[0].answers)

    @classmethod
    def n_targets(cls) -> int:
        return len(cls.TARGETS)
