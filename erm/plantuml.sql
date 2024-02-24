@startuml
!theme vibrant
!define TABLE(text) class text << (T,orchid) >>
!define PRIMARY_KEY(text) <u><b>text</b></u>
!define FOREIGN_KEY(text) <color:RoyalBlue><b>text</b></color>

TABLE(Account) {
    PRIMARY_KEY(account_id : int)
    FOREIGN_KEY(district_id : int)
    frequency : varchar
    date : int
}

TABLE(Card) {
    PRIMARY_KEY(card_id : int)
    FOREIGN_KEY(disp_id : int)
    type : varchar
    issued : datetime
}

TABLE(Client) {
    PRIMARY_KEY(client_id : int)
    birth_number : int
    FOREIGN_KEY(district_id : int)
}

TABLE(Disp) {
    PRIMARY_KEY(disp_id : int)
    FOREIGN_KEY(client_id : int)
    FOREIGN_KEY(account_id : int)
    type : varchar
}


TABLE(District) {
    PRIMARY_KEY(A1 : int) // District ID
    A2 : varchar // District name
    A3 : varchar // Region
    A4 : int // Total population
    A5 : int // Number of municipalities with < 500 inhabitants
    A6 : int // Number of municipalities with 500-1999 inhabitants
    A7 : int // Number of municipalities with 2000-9999 inhabitants
    A8 : int // Number of municipalities with >= 10000 inhabitants
    A9 : int // Number of cities
    A10 : float // Ratio of urban inhabitants
    A11 : int // Average salary
    A12 : float // Unemployment rate '95
    A13 : float // Unemployment rate '96
    A14 : int // Number of entrepreneurs per 1000 inhabitants
    A15 : int // Number of committed crimes '95
    A16 : int // Number of committed crimes '97
}
@enduml

TABLE(Loan) {
    PRIMARY_KEY(loan_id : int)
    FOREIGN_KEY(account_id : int)
    date : int
    amount : int
    duration : int
    payments : decimal
    status : char
}

TABLE(Order) {
    PRIMARY_KEY(order_id : int)
    FOREIGN_KEY(account_id : int)
    bank_to : char
    account_to : int
    amount : decimal
    k_symbol : varchar
}

TABLE(Trans) {
    PRIMARY_KEY(trans_id : int)
    FOREIGN_KEY(account_id : int)
    date : int
    type : varchar
    operation : varchar
    amount : decimal
    balance : decimal
    k_symbol : varchar
    bank : char
    account : int
}

Account --{ District
Disp -- Account
Disp -- Client
Card -- Disp
Loan -- Account
Order -- Account
Trans -- Account
@enduml