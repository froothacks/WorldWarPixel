# ruff: noqa: RUF001
"""Per-country, per-sex name pools so every agent gets a plausible everyday name.

Keys are the short country names used in ``worldwarpixel/data/wpp2024_population.csv``; countries
without their own pool fall back to a pool for their UN subregion. Each entry is
``(male given names, female given names, family names)`` as space-separated strings.
"""

import random

_POOLS: dict[str, tuple[str, str, str]] = {
    "India": (
        "Arjun Rohan Vikram Aditya Rahul Karan Suresh Anil",
        "Priya Ananya Kavya Meera Divya Sneha Pooja Lakshmi",
        "Sharma Patel Iyer Nair Gupta Reddy Singh Das Mehta Yadav Kumar Joshi",
    ),
    "China": (
        "Wei_Jun Hao Bo Lei Qiang Tao Feng",
        "Na Xiu Mei Yan Ling Fang Jing Li",
        "Zhang Wang Li Liu Chen Yang Huang Zhao Wu Zhou Xu Sun",
    ),
    "United States": (
        "James Marcus Tyler Kevin Brandon Michael Chris Daniel",
        "Emily Ashley Jasmine Maria Hannah Sarah Jessica Megan",
        "Miller Johnson Garcia Williams Nguyen Davis Rodriguez Smith Lee Brown Jones Taylor",
    ),
    "Indonesia": (
        "Budi Agus Putra Eko Dian Rizky Andi Hendra",
        "Siti Dewi Rina Ayu Fitri Indah Sari Lestari",
        "Rahma Santoso Wijaya Saputra Pratama Hidayat Kusuma Nugroho Setiawan Susanto",
    ),
    "Pakistan": (
        "Ahmed Bilal Usman Hamza Imran Ali Faisal Tariq",
        "Ayesha Fatima Zainab Hira Sana Maryam Amna Sadia",
        "Khan Malik Hussain Raza Sheikh Qureshi Butt Chaudhry Baig Siddiqui",
    ),
    "Nigeria": (
        "Chinedu Emeka Tunde Ibrahim Musa Segun Obinna Yusuf",
        "Ngozi Aisha Funke Chioma Blessing Halima Amaka Zainab",
        "Okafor Adeyemi Bello Okonkwo Abubakar Eze Adebayo Lawal Nwosu Mohammed",
    ),
    "Brazil": (
        "João Lucas Pedro Gabriel Rafael Mateus Bruno Thiago",
        "Ana Maria Camila Larissa Beatriz Juliana Fernanda Letícia",
        "Silva Santos Oliveira Souza Pereira Costa Rodrigues Almeida Ferreira Carvalho",
    ),
    "Bangladesh": (
        "Rahim Kamal Sohel Arif Hasan Jamal Rakib Tanvir",
        "Nusrat Sumaiya Tania Farhana Shirin Rima Mitu Sadia",
        "Hossain Islam Ahmed Rahman Chowdhury Akter Miah Khan Uddin Sarkar",
    ),
    "Russia": (
        "Dmitri Sergei Alexei Ivan Nikolai Andrei Maxim Pavel",
        "Olga Anastasia Elena Natalia Svetlana Irina Tatiana Ekaterina",
        "Ivanov Smirnov Kuznetsov Popov Volkov Sokolov Petrov Novikov Morozov Fedorov",
    ),
    "Ethiopia": (
        "Abebe Dawit Tesfaye Yonas Kebede Solomon Getachew Biruk",
        "Almaz Hana Meron Selam Tigist Bethlehem Marta Sara",
        "Tadesse Haile Bekele Alemu Girma Mekonnen Desta Assefa Tesfaye Worku",
    ),
    "Mexico": (
        "Carlos Luis Miguel Diego Alejandro José Fernando Javier",
        "Lupita Valeria Sofía Fernanda Ximena Guadalupe Daniela Mariana",
        "Hernández García Martínez López González Pérez Ramírez Sánchez Torres Flores",
    ),
    "Japan": (
        "Haruto Yuto Sota Ren Kaito Riku Daiki Takumi",
        "Yui Aoi Hina Sakura Mio Rin Yuna Hana",
        "Sato Suzuki Takahashi Tanaka Watanabe Ito Yamamoto Nakamura Kobayashi Kato",
    ),
    "Egypt": (
        "Ahmed Mohamed Omar Youssef Karim Mahmoud Mostafa Hassan",
        "Nour Mariam Salma Yasmin Hana Mona Dina Fatma",
        "Hassan Ibrahim Mahmoud Saleh Fathy Abdel-Rahman Mansour Farouk Shalaby El-Sayed",
    ),
    "Philippines": (
        "Jose Juan Mark Carlo Paolo Joshua Rafael Miguel",
        "Maria Angel Jenny Grace Rose Joy Kristine Angelica",
        "Santos Reyes Cruz Bautista Garcia Mendoza Ocampo Dela_Cruz Villanueva Ramos",
    ),
    "DR Congo": (
        "Patrick Didier Joseph Emmanuel Jean Christian Serge Papy",
        "Esther Chantal Nadine Grace Mireille Josephine Christelle Bijou",
        "Kabila Mukendi Ilunga Tshisekedi Kasongo Mbuyi Ngoy Lumumba Kalala Mwamba",
    ),
    "Vietnam": (
        "Minh Huy Duc Quang Tuan Nam Hoang Khanh",
        "Linh Anh Thu Mai Lan Huong Ngoc Phuong",
        "Nguyen Tran Le Pham Hoang Vu Dang Bui Do Ngo",
    ),
    "Iran": (
        "Reza Ali Amir Hossein Mehdi Mohammad Saeed Hamid",
        "Sara Niloufar Zahra Maryam Leila Fatemeh Neda Shirin",
        "Ahmadi Hosseini Karimi Rezaei Moradi Mousavi Jafari Rahimi Sadeghi Ebrahimi",
    ),
    "Turkey": (
        "Mehmet Ahmet Mustafa Emre Can Burak Murat Ali",
        "Elif Zeynep Ayşe Deniz Selin Merve Fatma Esra",
        "Yılmaz Kaya Demir Çelik Şahin Yıldız Aydın Öztürk Arslan Doğan",
    ),
    "Germany": (
        "Lukas Leon Felix Jonas Paul Max Tim Niklas",
        "Anna Lena Mia Sophie Hannah Laura Lisa Julia",
        "Müller Schmidt Schneider Fischer Weber Meyer Wagner Becker Hoffmann Koch",
    ),
    "Thailand": (
        "Somchai Anan Prasert Niran Kittisak Thanawat Chai Arthit",
        "Suda Malee Pim Ploy Nok Kanya Siriporn Wipa",
        "Saetang Srisuk Chaiyaporn Wongsawat Boonmee Rattanakul Thongchai Prasert Sukhum Khamsri",
    ),
    "United Kingdom": (
        "Oliver Jack Harry George Tom James Liam Callum",
        "Olivia Amelia Emily Sophie Charlotte Grace Ellie Chloe",
        "Smith Jones Taylor Brown Williams Wilson Evans Thomas Patel Khan Davies Murphy",
    ),
    "France": (
        "Louis Hugo Lucas Antoine Mathis Nathan Thomas Julien",
        "Camille Léa Chloé Manon Inès Louise Emma Sarah",
        "Martin Bernard Dubois Thomas Robert Petit Durand Leroy Moreau Simon",
    ),
    "Italy": (
        "Marco Luca Alessandro Matteo Andrea Francesco Giuseppe Davide",
        "Giulia Sofia Chiara Francesca Martina Sara Alessia Elena",
        "Rossi Russo Ferrari Esposito Bianchi Romano Colombo Ricci Marino Greco",
    ),
    "Spain": (
        "Alejandro Pablo Javier Daniel Álvaro Carlos Sergio Adrián",
        "Lucía María Paula Carmen Laura Marta Sara Elena",
        "García Martínez López Sánchez Pérez Gómez Fernández Ruiz Díaz Moreno",
    ),
    "Tanzania": (
        "Juma Baraka Hamisi Emmanuel Joseph Said Rashid Peter",
        "Amina Neema Zawadi Rehema Grace Fatuma Asha Happiness",
        "Mwangi Komba Mushi Kimaro Mrema Shayo Lyimo Mapunda Kessy Mollel",
    ),
    "South Africa": (
        "Sipho Thabo Bongani Lerato Johan Pieter Sibusiso Themba",
        "Thandiwe Nomvula Zanele Ayanda Lindiwe Annelie Precious Naledi",
        "Dlamini Nkosi Mokoena Botha Van_Wyk Ndlovu Khumalo Naidoo Pillay Zulu",
    ),
    "Kenya": (
        "Kamau Otieno Kipchoge Brian Dennis Mwangi Omondi Kiprop",
        "Wanjiru Akinyi Chebet Faith Mercy Njeri Atieno Wairimu",
        "Kamau Odhiambo Kiprotich Mwangi Wanjiku Ochieng Njoroge Karanja Mutua Kibet",
    ),
    "Myanmar": (
        "Aung Kyaw Zaw Min Htet Thura Naing Soe",
        "Su Thandar Hnin Khin Aye Nwe Ei Thiri",
        "Win Lin Tun Oo Htun Myint Hlaing Thein Maung Lwin",
    ),
    "South Korea": (
        "Min-jun Seo-jun Ji-ho Do-yun Jun-seo Hyun-woo Ji-hoon Sung-min",
        "Seo-yeon Ji-woo Ha-eun Min-seo Ji-yoon Soo-ah Ye-jin Eun-ji",
        "Kim Lee Park Choi Jung Kang Cho Yoon Jang Lim",
    ),
    "Colombia": (
        "Santiago Andrés Juan Camilo Sebastián Carlos Felipe Mateo",
        "Valentina Camila Paula Daniela Laura Juliana Carolina Natalia",
        "Rodríguez Gómez González Martínez García López Hernández Díaz Ramírez Castro",
    ),
    "Uganda": (
        "Moses Ronald Isaac Brian Joshua Ivan Patrick Derrick",
        "Sarah Esther Doreen Grace Ruth Joan Patience Shamim",
        "Okello Mukasa Ssemakula Nakato Kato Namukasa Ochieng Byaruhanga Tumwine Kiprotich",
    ),
    "Sudan": (
        "Mohamed Ahmed Omar Osman Abdallah Khalid Hassan Mustafa",
        "Fatima Amira Maha Samia Nadia Hiba Rania Awadia",
        "Ahmed Osman Abdalla Ibrahim Hamid Elamin Mohamed Idris Bashir Taha",
    ),
    "Algeria": (
        "Mohamed Karim Yacine Amine Riad Sofiane Walid Rachid",
        "Amina Lina Sarah Meriem Yasmine Imane Nour Samira",
        "Benali Boudiaf Khelifi Brahimi Saidi Mansouri Hamidi Bouazza Cherif Belkacem",
    ),
    "Iraq": (
        "Ali Hussein Mohammed Ahmed Omar Hassan Mustafa Karrar",
        "Zahra Fatima Noor Zainab Hawraa Maryam Rusul Sara",
        "Al-Jabouri Al-Saadi Al-Maliki Al-Tamimi Al-Dulaimi "
        "Al-Shammari Al-Zubaidi Hassan Kadhim Abbas",
    ),
    "Afghanistan": (
        "Ahmad Mohammad Abdul Hamid Farid Naser Zabi Rahim",
        "Fatima Zahra Mariam Laila Nadia Farida Shabnam Roya",
        "Ahmadi Rahimi Karimi Hashimi Noori Sultani Haidari Safi Akbari Nazari",
    ),
    "Yemen": (
        "Ali Mohammed Ahmed Saleh Abdullah Yahya Khalid Omar",
        "Amal Fatima Samira Huda Noor Mariam Salma Asma",
        "Al-Hamdani Al-Shami Al-Sanani Al-Qadi Saleh Al-Hadi Al-Ahdal Mansour Al-Maqtari Naji",
    ),
    "Uzbekistan": (
        "Jasur Bekzod Sardor Alisher Rustam Jahongir Timur Otabek",
        "Nilufar Madina Dilnoza Gulnora Zarina Sevara Malika Feruza",
        "Karimov Rashidov Tashkentov Yusupov Mirzaev Abdullaev Saidov Ismoilov Usmonov Nazarov",
    ),
    "Ukraine": (
        "Oleksandr Andriy Dmytro Taras Bohdan Mykola Yuriy Ivan",
        "Oksana Kateryna Iryna Olena Natalia Tetiana Anna Yulia",
        "Shevchenko Kovalenko Bondarenko Boyko Tkachenko Kravchenko Melnyk Koval Lysenko Rudenko",
    ),
    "Saudi Arabia": (
        "Abdullah Mohammed Fahad Saud Khalid Faisal Turki Nasser",
        "Noura Sara Reem Lama Hessa Maha Aljohara Dana",
        "Al-Qahtani Al-Otaibi Al-Ghamdi Al-Harbi Al-Dosari "
        "Al-Shehri Al-Zahrani Al-Mutairi Al-Anazi Al-Subaie",
    ),
    "Poland": (
        "Jakub Kacper Mateusz Piotr Tomasz Marcin Michał Krzysztof",
        "Zuzanna Julia Maja Anna Katarzyna Agnieszka Magdalena Natalia",
        "Nowak Kowalski Wiśniewski Wójcik Kowalczyk "
        "Kamiński Lewandowski Zieliński Szymański Woźniak",
    ),
    "Peru": (
        "José Luis Carlos Jorge Miguel Diego Alonso Renato",
        "María Rosa Ana Lucía Carmen Fiorella Milagros Valeria",
        "Quispe Flores Huamán García Rodríguez Mamani Sánchez Torres Chávez Rojas",
    ),
    "Malaysia": (
        "Ahmad Muhammad Amir Hafiz Wei_Jun Faiz Arjun Daniel",
        "Nur Aisyah Siti Farah Mei_Ling Priya Hui_Ying Aina",
        "Abdullah Ismail Tan Lim Lee Rahman Wong Raj Hassan Yusof",
    ),
    "Mozambique": (
        "Carlos António José Paulo Fernando Armando Tomás Jorge",
        "Ana Maria Joana Isabel Lúcia Teresa Rosa Felismina",
        "Machel Sitoe Mondlane Cossa Chissano Mabunda Nhantumbo Matsinhe Macuácua Tembe",
    ),
    "Morocco": (
        "Youssef Mohamed Omar Amine Hamza Rachid Karim Mehdi",
        "Fatima Khadija Salma Imane Sara Meryem Hajar Nadia",
        "Alaoui Benjelloun El_Idrissi Tazi Bennani Amrani Berrada Fassi Chraibi Lahlou",
    ),
    "Ghana": (
        "Kwame Kofi Kwesi Yaw Kojo Nana Emmanuel Samuel",
        "Ama Akosua Abena Adwoa Efua Akua Esi Yaa",
        "Mensah Osei Boateng Owusu Asante Appiah Darko Agyemang Acheampong Ofori",
    ),
    "Angola": (
        "João Manuel António Pedro Domingos Francisco José Miguel",
        "Maria Ana Isabel Josefa Luísa Teresa Domingas Paula",
        "Dos_Santos Fernandes Gomes Da_Silva Manuel Francisco Domingos Neto Baptista Lopes",
    ),
    "Canada": (
        "Liam Noah Ethan Lucas Benjamin Olivier Nathan Owen",
        "Emma Olivia Sophia Chloé Ava Charlotte Léa Zoe",
        "Smith Tremblay Roy Brown Wilson Gagnon Lee Singh Côté Martin",
    ),
    "Argentina": (
        "Santiago Mateo Juan Nicolás Facundo Martín Lucas Agustín",
        "Sofía Valentina Camila Martina Lucía Florencia Agustina Julieta",
        "González Rodríguez Gómez Fernández López Díaz Martínez Pérez Romero Sosa",
    ),
    "Nepal": (
        "Bikash Suresh Ramesh Prakash Sanjay Dipesh Anil Rajan",
        "Sita Sunita Anita Gita Pooja Sabina Sarita Laxmi",
        "Shrestha Gurung Tamang Rai Thapa Magar Adhikari Karki Lama Maharjan",
    ),
    "Australia": (
        "Jack Oliver William Lachlan Noah Liam Cooper Ethan",
        "Charlotte Olivia Mia Amelia Chloe Isla Ruby Zoe",
        "Smith Jones Williams Brown Wilson Taylor Nguyen Lee Martin Thompson",
    ),
    "Venezuela": (
        "José Luis Carlos Jesús Miguel Rafael Andrés Daniel",
        "María Carmen Ana Yuleidy Gabriela Daniela Andrea Yolanda",
        "González Rodríguez Hernández Pérez García Martínez Rojas Díaz Torres Ramírez",
    ),
    "Netherlands": (
        "Daan Sem Lucas Bram Thijs Jesse Finn Luuk",
        "Emma Julia Sophie Lotte Tess Fleur Anna Sara",
        "De_Jong Jansen De_Vries Van_den_Berg Bakker Visser Smit Meijer Mulder Bos",
    ),
    "Cameroon": (
        "Samuel Paul Jean Emmanuel Serge Patrick Joseph Aboubakar",
        "Marie Christelle Aïcha Brigitte Josiane Nadège Sandrine Fadimatou",
        "Mbarga Nkomo Fotso Tchoupo Ngono Biya Kamga Essomba Nana Abanda",
    ),
    "Niger": (
        "Abdou Moussa Ibrahim Hassane Mahamadou Issoufou Boubacar Ousmane",
        "Aïcha Fatima Mariama Hadiza Zeinabou Rakia Aminatou Haoua",
        "Issoufou Moussa Abdou Mahamadou Hassane Oumarou Boubacar Garba Yacouba Saidou",
    ),
    "Sri Lanka": (
        "Nuwan Kasun Chaminda Dilshan Lahiru Ruwan Sanjeewa Tharindu",
        "Dilani Nadeesha Sachini Hiruni Kumari Thilini Nilmini Dinusha",
        "Perera Fernando Silva Jayasinghe Bandara "
        "Wickramasinghe Gunawardena Rajapaksa Dissanayake Herath",
    ),
    "Madagascar": (
        "Andry Hery Tahina Mamy Rija Faly Toky Lova",
        "Mialy Fanja Hanta Vola Soa Lalaina Nirina Tiana",
        "Rakotoarisoa Randriamanana Rasolofo Andrianarivo Razafindrakoto "
        "Ramanantsoa Rabe Rajaonarison Rakoto Raharison",
    ),
    "Côte d'Ivoire": (
        "Kouassi Yao Koffi Konan Kouamé Ibrahim Seydou Adama",
        "Aya Akissi Affoué Adjoua Mariam Fatou Aminata Awa",
        "Kouassi Koné Ouattara Yao Konan Traoré Diabaté Bamba Kouamé Coulibaly",
    ),
    "North Korea": (
        "Chol Myong Hyok Song Il Jun Yong Kwang",
        "Sun Hui Yong-mi Ok Mi-ran Jong Hwa Un-ju Kyong",
        "Kim Ri Pak Choe Jong Kang Han Jo Yun Ryu",
    ),
    "Burkina Faso": (
        "Issa Boukary Adama Moussa Ousmane Souleymane Rasmane Idrissa",
        "Awa Mariam Fatimata Aïcha Salimata Aminata Rasmata Assetou",
        "Ouédraogo Sawadogo Kaboré Compaoré Zongo Traoré Ilboudo Sankara Nikiéma Kafando",
    ),
    "Mali": (
        "Moussa Mamadou Ibrahim Oumar Seydou Boubacar Modibo Amadou",
        "Aminata Fatoumata Kadiatou Mariam Awa Oumou Assa Djeneba",
        "Traoré Diarra Coulibaly Keita Sidibé Touré Konaté Cissé Diallo Doumbia",
    ),
    "Syria": (
        "Ahmad Mohammad Omar Khaled Yousef Samer Firas Bashar",
        "Rana Hala Nour Lina Reem Maya Dima Salma",
        "Al-Ahmad Haddad Khoury Al-Hassan Nassar Saleh Al-Ali Khalil Azzam Darwish",
    ),
    "Kazakhstan": (
        "Arman Nursultan Daniyar Yerlan Bekzat Aibek Timur Askar",
        "Aigerim Dana Aizhan Madina Gulnara Assel Zhanar Saule",
        "Nazarbayev Akhmetov Omarov Suleimenov Ospanov Abenov Zhumabekov Bekov Serikov Kasymov",
    ),
    "Chile": (
        "Matías Benjamín Vicente Sebastián Diego Cristóbal Tomás Felipe",
        "Sofía Martina Isidora Antonia Catalina Valentina Fernanda Javiera",
        "González Muñoz Rojas Díaz Pérez Soto Contreras Silva Martínez Sepúlveda",
    ),
    "Romania": (
        "Andrei Alexandru Mihai Ionuț Ștefan Cristian Bogdan Florin",
        "Maria Andreea Elena Ioana Alexandra Ana Cristina Gabriela",
        "Popescu Ionescu Popa Radu Dumitru Stan Stoica Gheorghe Constantin Marin",
    ),
}

_REGIONAL: dict[str, tuple[str, str, str]] = {
    "Western Africa": (
        "Kwame Moussa Ibrahim Kofi Adama Sekou Amadou Emmanuel",
        "Ama Fatou Aminata Mariama Adwoa Aïssatou Grace Khadija",
        "Mensah Diallo Touré Osei Camara Sow Keita Bah Cissé Johnson",
    ),
    "Eastern Africa": (
        "Juma Emmanuel Joseph Peter Hassan Daniel Moses Samuel",
        "Amina Grace Neema Faith Esther Mercy Zawadi Rehema",
        "Mwangi Ochieng Hassan Nyerere Kamau Abdi Mohamed Mugisha Omar Kimani",
    ),
    "Middle Africa": (
        "Jean Patrick Emmanuel Joseph Paul Christian Serge Dieudonné",
        "Marie Chantal Esther Grâce Josiane Nadège Christelle Brigitte",
        "Nguema Mbarga Kabongo Ndong Okemba Mavoungou Ilunga Essono Ngoma Mbeki",
    ),
    "Southern Africa": (
        "Sipho Thabo Tebogo Kagiso Johan Bongani Themba Lesego",
        "Thandiwe Lerato Naledi Nomsa Zanele Precious Annelie Ayanda",
        "Dlamini Nkosi Moyo Botha Ndlovu Molefe Khumalo Van_Wyk Sithole Mokoena",
    ),
    "Northern Africa": (
        "Mohamed Ahmed Omar Youssef Karim Ali Hassan Amine",
        "Fatima Amina Nour Salma Mariam Yasmine Sara Hiba",
        "Ben_Ali Hassan Ibrahim Mansour Saleh Osman Trabelsi Benali Gaddafi El-Sayed",
    ),
    "Western Asia": (
        "Ali Mohammed Ahmad Omar Khalid Yousef Hassan Sami",
        "Fatima Noor Mariam Sara Layla Rana Hala Reem",
        "Haddad Al-Sayed Khalil Nasser Saleh Hassan Mansour Al-Ahmad Khoury Aziz",
    ),
    "Central Asia": (
        "Arman Rustam Timur Bekzod Daniyar Alisher Farrukh Aibek",
        "Aigerim Madina Nilufar Dilnoza Zarina Aizhan Gulnara Malika",
        "Karimov Akhmetov Rashidov Omarov Saidov Yusupov Nazarov Mirzoev Abdullaev Bekov",
    ),
    "Southern Asia": (
        "Arjun Rahul Ahmed Bikash Nuwan Ali Suresh Imran",
        "Priya Fatima Sita Ayesha Dilani Meera Anita Zainab",
        "Sharma Khan Shrestha Perera Singh Hossain Patel Gurung Ahmed Das",
    ),
    "Eastern Asia": (
        "Wei Haruto Min-jun Jun Batbayar Hao Ren Ji-ho",
        "Mei Yui Seo-yeon Ling Sarnai Na Sakura Ji-woo",
        "Wang Sato Kim Li Zhang Tanaka Park Chen Suzuki Batbold",
    ),
    "South-Eastern Asia": (
        "Minh Budi Somchai Jose Aung Ahmad Sok Boun",
        "Linh Siti Suda Maria Su Nur Sopheap Noy",
        "Nguyen Santoso Santos Tan Win Lim Sok Phan Wijaya Reyes",
    ),
    "Western Europe": (
        "Lukas Louis Noah Hugo Finn Jonas Luca Léon",
        "Emma Léa Anna Mia Chloé Sophie Lena Julia",
        "Müller Martin Dubois Schmidt Peeters Meier Bernard Weber Janssens Huber",
    ),
    "Northern Europe": (
        "Oliver William Lucas Noah Liam Emil Oskar Elias",
        "Emma Olivia Alice Maja Ella Sofia Freya Ida",
        "Andersen Johansson Hansen Nielsen Smith Larsen Virtanen Olsen Murphy Berg",
    ),
    "Southern Europe": (
        "Marco Alejandro Luka Nikola Luís Giorgos Andrej Pablo",
        "Giulia Lucía Ana Marija Sofia Maria Elena Ivana",
        "Rossi García Horvat Papadopoulos Silva Novak Jovanović Kovač Ferreira Russo",
    ),
    "Eastern Europe": (
        "Andrei Jakub Dmitri Ivan Mykola Péter Georgi Pavel",
        "Anna Olga Maria Elena Katarzyna Iryna Eva Natalia",
        "Novák Kowalski Ivanov Popescu Horváth Petrov Shevchenko Dimitrov Nagy Kovács",
    ),
    "Northern America": (
        "James Liam Noah Michael Ethan Daniel Lucas Owen",
        "Emily Emma Olivia Sarah Ava Hannah Sophia Chloe",
        "Smith Johnson Williams Brown Tremblay Garcia Miller Lee Wilson Martin",
    ),
    "Caribbean": (
        "José Jean Marcus Luis Pierre Andre Carlos Devon",
        "María Marie Keisha Ana Nadège Shanice Rosa Daniela",
        "Rodríguez Jean-Baptiste Pérez Williams Pierre Brown Martínez Louis García Campbell",
    ),
    "Central America": (
        "José Carlos Luis Juan Miguel Óscar Manuel Jorge",
        "María Ana Rosa Carmen Sofía Daniela Alejandra Gabriela",
        "Hernández López García Martínez Pérez Rodríguez González Ramírez Castillo Morales",
    ),
    "South America": (
        "José Carlos Luis Juan Diego Santiago Mateo Andrés",
        "María Ana Camila Valentina Sofía Lucía Daniela Gabriela",
        "González Rodríguez Silva Pérez López Fernández Gómez Martínez Rojas Díaz",
    ),
    "Australia/New Zealand": (
        "Jack Oliver Liam Noah Lachlan William Cooper Nikau",
        "Charlotte Olivia Mia Amelia Isla Aria Ruby Chloe",
        "Smith Jones Williams Brown Wilson Taylor Walker Thompson Tane Parata",
    ),
    "Melanesia": (
        "John Peter Joseph Michael Ratu Samuel Isikeli Kila",
        "Mary Grace Ruth Elizabeth Adi Ana Sera Litia",
        "Kaupa Tabua Natuman Vuki Namaliu Waqa Somare Kila Rabuka Sope",
    ),
    "Micronesia": (
        "John Joseph Peter Michael David Kaleb Isaac Jacob",
        "Mary Maria Rose Ana Grace Lydia Sarah Ruth",
        "Nakamura Santos Mori Falcam Remengesau Heine Hadley Peter Tomeing Panuelo",
    ),
    "Polynesia": (
        "Sione Tevita Manu Viliami Tama Sefo Lio Rangi",
        "Mele Ana Lupe Sina Moana Fetu Tui Hina",
        "Tupou Taufa Fifita Havili Teariki Vaea Ioane Tuilagi Faumuina Tamati",
    ),
}

Pool = tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]


def _parse(*groups: str) -> Pool:
    """Space-separated names; an underscore joins multi-word names ("Van_Wyk")."""
    m, f, s = (tuple(n.replace("_", " ") for n in g.split()) for g in groups)
    return m, f, s


NAME_POOLS: dict[str, Pool] = {k: _parse(m, f, s) for k, (m, f, s) in _POOLS.items()}
REGIONAL_POOLS: dict[str, Pool] = {k: _parse(m, f, s) for k, (m, f, s) in _REGIONAL.items()}


def random_name(
    country: str,
    rng: random.Random,
    used: set[str],
    sex: int | None = None,
    subregion: str | None = None,
) -> str:
    """A unique full name for ``country`` (``sex``: 1 man, 2 woman, None either)."""
    pool = NAME_POOLS.get(country) or REGIONAL_POOLS.get(subregion or "")
    if pool is None:
        pool = REGIONAL_POOLS["Northern America"]
    males, females, lasts = pool
    firsts = males if sex == 1 else females if sex == 2 else males + females
    for _ in range(50):
        name = f"{rng.choice(firsts)} {rng.choice(lasts)}"
        if name not in used:
            used.add(name)
            return name
    name = f"{rng.choice(firsts)} {rng.choice(lasts)} {len(used) + 1}"
    used.add(name)
    return name
