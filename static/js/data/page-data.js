export const pageData = {
  page: {
    title: "FSU SC Artificial Intelligence Seminar",
    description: "Weekly conversations on AI, machine learning, scientific computing, and applied research."
  },
  navigation: {
    siteName: "FSU SC Artificial Intelligence Seminar",
    items: [
      { id: "home", label: "Home", href: "/", active: true },
      { id: "schedule", label: "Schedule", href: "/schedule/" },
      { id: "speakers", label: "Speakers", href: "/speakers/" },
      { id: "archive", label: "Archive", href: "/archive/" },
      { id: "community", label: "Community", href: "/#community" }
    ]
  },
  hero: {
    content: {
      headline: "FSU SC Artificial Intelligence Seminar",
      subtitle: "Weekly conversations on AI, machine learning, scientific computing, and applied research.",
      buttons: [
        { label: "View Schedule", href: "/schedule/", variant: "primary", icon: "calendar" },
        { label: "Join Discord", href: "https://discord.com/invite/raTxTXmM5B", variant: "discord", icon: "discord" },
        { label: "Join via Zoom", href: "https://fsu.zoom.us/j/9038137210", variant: "secondary-dark", icon: "zoom" },
        { label: "Join Mailing List", href: "mailto:gerlebacher@fsu.edu", variant: "outline-gold", icon: "mail" }
      ]
    },
    nextSeminarCard: {
      label: "Next Seminar",
      dateTime: "June 13, 2026 - 12:00 PM ET",
      locationLinks: [
        { label: "DSL/SC-499", href: "https://goo.gl/maps/BJLxE3Q7H1MTBqMu6" },
        { label: "Zoom", href: "https://fsu.zoom.us/j/9038137210" }
      ],
      talkTitle: "Building Reliable AI Workflows for Scientific Teams",
      speaker: "Brendon Gutierrez",
      affiliation: "Florida State University",
      speakerImage: "data/speaker-images/brendon_gutierrez.jpg"
    }
  },
  sections: [
    {
      id: "why-attend",
      intro: "A welcoming forum for students, researchers, and faculty to learn, share ideas, and build collaborations across disciplines.",
      sisterSeminar: {
        kicker: "Sister Seminar",
        text: "We run this series alongside the FSU Data Science Seminar, hosted by the Department of Mathematics.",
        link: {
          label: "Visit the Seminar",
          href: "https://sites.google.com/view/fsu-data-science-seminar/home"
        }
      },
      guideLink: {
        label: "Presenting at the seminar? Read our NASA Trichotomy guide",
        href: "/trichotemy.html"
      },
      cards: [
        {
          title: "Interdisciplinary Community",
          description: "Connect with researchers and practitioners from across FSU and beyond.",
          icon: "users"
        },
        {
          title: "Research Talks",
          description: "Hear cutting-edge research from leading experts in AI and related fields.",
          icon: "presentation"
        },
        {
          title: "Applied AI Topics",
          description: "Explore real-world applications in science, engineering, health, society, and more.",
          icon: "cpu"
        },
        {
          title: "Student & Faculty Collaboration",
          description: "Find opportunities to collaborate, learn, and grow together.",
          icon: "graduation-cap"
        }
      ]
    },
    {
      id: "schedule",
      action: { label: "View full schedule", href: "/schedule/" },
      items: []
    },
    {
      id: "speakers",
      action: { label: "View all speakers", href: "/speakers/" },
      speakers: [
        {
          name: "Dr. Jane Smith",
          title: "Associate Professor",
          department: "School of Computational Science and Engineering",
          affiliation: "Georgia Institute of Technology",
          topic: "AI & Science",
          website: "#speakers",
          image: ""
        },
        {
          name: "Dr. Carlos Rojas",
          title: "Assistant Professor",
          department: "Paul G. Allen School of Computer Science & Engineering",
          affiliation: "University of Washington",
          topic: "ML Systems",
          website: "#speakers",
          image: ""
        },
        {
          name: "Dr. Priya Natarajan",
          title: "Professor",
          department: "Department of Computer Science",
          affiliation: "Cornell University",
          topic: "AI Safety",
          website: "#speakers",
          image: ""
        }
      ]
    },
    {
      id: "community",
      items: [
        {
          title: "Discord",
          description: "Join our Discord server to connect, ask questions, and stay up to date.",
          icon: "message-circle",
          action: { label: "Join", href: "https://discord.com/invite/raTxTXmM5B" }
        },
        {
          title: "Mailing List",
          description: "Receive weekly updates about upcoming talks and events.",
          icon: "mail",
          action: { label: "Email", href: "mailto:gerlebacher@fsu.edu" }
        }
      ]
    }
  ],
  footer: {
    branding: {
      logoText: "FSU",
      department: "Scientific Computing",
      university: "Florida State University"
    },
    description:
      "The FSU SC Artificial Intelligence Seminar is organized by FSU Scientific Computing and the research community. The series brings together students, researchers, and faculty for weekly conversations on AI, machine learning, scientific computing, and applied research.",
    contact: {
      title: "Contact",
      items: [
        { label: "Mailing List Request", href: "mailto:gerlebacher@fsu.edu", icon: "mail" },
        { label: "Department Website", href: "https://www.sc.fsu.edu", icon: "globe" },
        { label: "Building Location", href: "https://goo.gl/maps/BJLxE3Q7H1MTBqMu6", icon: "map-pin" },
        { label: "Zoom Room", href: "https://fsu.zoom.us/j/9038137210", icon: "zoom" }
      ]
    },
    social: {
      title: "Follow & Connect",
      items: [
        { label: "Discord", href: "https://discord.com/invite/raTxTXmM5B", icon: "discord" },
        {
          label: "LinkedIn",
          href: "https://www.linkedin.com/school/florida-state-university---department-of-scientific-computing/",
          icon: "linkedin"
        }
      ]
    }
  }
};
