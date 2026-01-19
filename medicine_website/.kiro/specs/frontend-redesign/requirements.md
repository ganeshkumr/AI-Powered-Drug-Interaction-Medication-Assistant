# Frontend Redesign Requirements

## Introduction

This spec outlines the complete redesign of the Medicine Assistant frontend to create a professional, user-friendly, and accessible healthcare application. The redesign focuses on improving UX, visual design, and user flow while maintaining all existing backend functionality.

## Glossary

- **System**: The Medicine Assistant React frontend application
- **User**: Patient or healthcare consumer using the application
- **Quick Check**: Fast drug interaction check without login
- **Profile Check**: Personalized check using user's health profile
- **Medication Wallet**: User's saved medications list
- **Risk Score**: GNN-predicted interaction risk percentage (0-100)

## Requirements

### Requirement 1: Professional Visual Design System

**User Story:** As a user, I want a modern, professional interface that builds trust and is easy to use.

#### Acceptance Criteria

1. THE System SHALL implement the teal primary color (#2EA79B) for all primary actions and branding
2. THE System SHALL use amber accent color (#F4B400) for warnings and highlights
3. THE System SHALL use status colors: green (#16A34A) for safe, yellow (#F59E0B) for caution, red (#EF4444) for danger
4. THE System SHALL use Inter font for body text at 16-18px size
5. THE System SHALL use Poppins font for headings

### Requirement 2: Accessible and Responsive Layout

**User Story:** As a user with accessibility needs, I want the interface to be keyboard navigable and screen-reader friendly.

#### Acceptance Criteria

1. THE System SHALL ensure all interactive elements are minimum 44px in height and width
2. THE System SHALL provide ARIA labels for all interactive elements
3. THE System SHALL use color plus icon plus text for all status indicators
4. THE System SHALL be fully keyboard navigable
5. THE System SHALL provide focus indicators with 2px outline

### Requirement 3: Quick Check Flow (No Login Required)

**User Story:** As a new user, I want to quickly check drug interactions without creating an account.

#### Acceptance Criteria

1. THE System SHALL display a landing page with "Check Interaction" CTA
2. WHEN user clicks "Check Interaction", THE System SHALL open a modal for drug entry
3. THE System SHALL provide autocomplete drug search
4. THE System SHALL allow adding multiple drugs as chips
5. THE System SHALL display risk analysis results without requiring login

### Requirement 4: Enhanced Navigation

**User Story:** As a user, I want clear navigation to access all features easily.

#### Acceptance Criteria

1. THE System SHALL display a sticky navbar with logo, status, and profile
2. THE System SHALL show "AI-HealthMate" branding with pill icon
3. THE System SHALL display Google Fit connection status in navbar
4. THE System SHALL provide profile dropdown with Settings, Privacy, Logout
5. THE System SHALL include a footer with About, Quick Links, and Legal sections

### Requirement 5: Improved Risk Analysis Display

**User Story:** As a user, I want to clearly understand the risk level and why it exists.

#### Acceptance Criteria

1. THE System SHALL display risk score as a radial gauge (0-100)
2. THE System SHALL show risk level as SAFE/CAUTION/DANGEROUS with color coding
3. THE System SHALL provide "Explain Why" button for detailed analysis
4. THE System SHALL display AI explanation in plain language
5. THE System SHALL show GNN prediction percentage prominently

### Requirement 6: Medication Wallet

**User Story:** As a user, I want to save safe medications and manage them easily.

#### Acceptance Criteria

1. WHEN risk is SAFE, THE System SHALL provide "Save to Medication Wallet" button
2. THE System SHALL display saved medications as cards with dose and schedule
3. THE System SHALL allow editing and deleting saved medications
4. THE System SHALL show medication history and past checks
5. THE System SHALL provide reminder setup for each medication

### Requirement 7: Enhanced Profile Management

**User Story:** As a user, I want to manage my health profile with progressive disclosure.

#### Acceptance Criteria

1. THE System SHALL provide a single-column profile form
2. THE System SHALL use progressive disclosure for optional fields
3. THE System SHALL display conditions as multi-select chips
4. THE System SHALL provide inline validation and hints
5. THE System SHALL show Google Fit connection toggle

### Requirement 8: Smooth Animations and Micro-interactions

**User Story:** As a user, I want smooth, delightful interactions that provide feedback.

#### Acceptance Criteria

1. THE System SHALL animate page transitions with fade-in effects
2. THE System SHALL show loading states with spinner and microcopy
3. THE System SHALL animate risk gauge with pulse effect
4. THE System SHALL provide "pill drop" animation when saving medications
5. THE System SHALL show toast notifications for actions

### Requirement 9: Mobile-First Responsive Design

**User Story:** As a mobile user, I want the app to work perfectly on my phone.

#### Acceptance Criteria

1. THE System SHALL use mobile-first responsive design
2. THE System SHALL stack columns on mobile devices
3. THE System SHALL provide touch-friendly tap targets
4. THE System SHALL optimize font sizes for mobile readability
5. THE System SHALL hide/collapse secondary content on small screens

### Requirement 10: Trust and Privacy Indicators

**User Story:** As a user, I want to know my health data is secure and private.

#### Acceptance Criteria

1. THE System SHALL display "Health data processed locally" message in footer
2. THE System SHALL show consent checkbox for Google Fit data usage
3. THE System SHALL provide "What data we read" modal for transparency
4. THE System SHALL display last sync time for Google Fit
5. THE System SHALL show privacy policy and terms links

### Requirement 11: Dashboard Feature Separation

**User Story:** As a logged-in user, I want a focused dashboard that shows only medication management features relevant to my authenticated session.

#### Acceptance Criteria

1. THE System SHALL NOT display Emergency Drug Check on the post-login dashboard
2. THE System SHALL NOT display Live Health Monitoring on the post-login dashboard
3. THE System SHALL display Add New Medication functionality on the post-login dashboard
4. THE System SHALL display Medication Wallet on the post-login dashboard
5. WHEN user is not logged in, THE System SHALL display Emergency Drug Check on the landing page
