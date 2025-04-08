#deneme1

import requests
import datetime
from datetime import timedelta
import random  # For simulating price estimations where real API isn't available
import heapq  # For Dijkstra's algorithm
import math   # For distance calculations
import copy   # For deep copying objects

# ---------------------- FOURSQUARE ---------------------- #
def get_foursquare_places(city, limit=5):
    url = "https://api.foursquare.com/v3/places/search"
    headers = {
        "accept": "application/json",
        "Authorization": "fsq3s5hEBcp08bAusKqLz+wyaMnksJ78tmvPMidEfAe+Ktw="
    }
    params = {
        "near": city,
        "limit": limit,
        "sort": "POPULARITY"
    }

    response = requests.get(url, headers=headers, params=params)
    data = response.json()
    print(f"🔹 {city} için popüler yerler:")
    for place in data.get("results", []):
        print("-", place["name"])
    print()

def get_foursquare_places_return(city, limit=5):
    """Returns a list of popular places in the city"""
    url = "https://api.foursquare.com/v3/places/search"
    headers = {
        "accept": "application/json",
        "Authorization": "fsq3s5hEBcp08bAusKqLz+wyaMnksJ78tmvPMidEfAe+Ktw="
    }
    params = {
        "near": city,
        "limit": limit,
        "sort": "POPULARITY"
    }

    try:
        response = requests.get(url, headers=headers, params=params)
        data = response.json()
        places = []
        for place in data.get("results", []):
            places.append(place["name"])
        return places
    except Exception as e:
        print(f"Foursquare API hatası: {str(e)}")
        return []


# ---------------------- OPEN-METEO ---------------------- #
def get_weather(city_lat, city_lon):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": city_lat,
        "longitude": city_lon,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
        "timezone": "auto"
    }

    response = requests.get(url, params=params)
    data = response.json()
    print("🌤️ Hava durumu tahmini (bugünden itibaren):")
    for i in range(3):
        date = data['daily']['time'][i]
        t_min = data['daily']['temperature_2m_min'][i]
        t_max = data['daily']['temperature_2m_max'][i]
        rain = data['daily']['precipitation_sum'][i]
        print(f"{date}: {t_min}°C - {t_max}°C, Yağış: {rain}mm")
    print()


# ---------------------- AMADEUS OTEL ---------------------- #
def get_amadeus_access_token(client_id, client_secret):
    url = "https://test.api.amadeus.com/v1/security/oauth2/token"
    headers = {
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = {
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret
    }

    response = requests.post(url, headers=headers, data=data)
    return response.json()["access_token"]


def get_hotels(city_code, access_token):
    url = "https://test.api.amadeus.com/v1/shopping/hotel-offers"
    headers = {
        "Authorization": f"Bearer {access_token}"
    }
    params = {
        "cityCode": city_code,
        "adults": 1,
        "roomQuantity": 1,
        "radius": 20,
        "radiusUnit": "KM",
        "paymentPolicy": "NONE",
        "includeClosed": False,
        "bestRateOnly": True,
        "view": "FULL",
        "sort": "PRICE"
    }

    try:
        response = requests.get(url, headers=headers, params=params)
        data = response.json()
        
        print("🏨 Uygun otel teklifleri:")
        
        # API yanıt yapısını kontrol edelim
        if "data" in data and len(data.get("data", [])) > 0:
            for offer in data.get("data", [])[:3]:
                try:
                    name = offer["hotel"]["name"]
                    price = offer["offers"][0]["price"]["total"]
                    currency = offer["offers"][0]["price"]["currency"]
                    print(f"- {name}: {price} {currency}")
                except KeyError as e:
                    print(f"- Otel bilgilerinde eksik alan: {str(e)}")
            print()
        else:
            if "errors" in data:
                print(f"Hata: {data['errors'][0]['title']} - {data['errors'][0]['detail']}")
            else:
                print(f"Bu şehir için uygun otel teklifi bulunamadı.")
            print()
    except Exception as e:
        print(f"Otel arama hatası: {str(e)}")
        print()


# ---------------------- AMADEUS ŞEHIR ARAMA ---------------------- #
def search_city(city_name, access_token):
    """Amadeus API üzerinden şehir arama"""
    url = "https://test.api.amadeus.com/v1/reference-data/locations"
    headers = {
        "Authorization": f"Bearer {access_token}"
    }
    params = {
        "keyword": city_name,
        "subType": "CITY",
        "page[limit]": 5
    }
    
    try:
        response = requests.get(url, headers=headers, params=params)
        data = response.json()
        
        if "data" in data and len(data["data"]) > 0:
            city_options = []
            print(f"\nŞehir araması sonuçları '{city_name}':")
            
            for i, location in enumerate(data["data"]):
                city_info = {
                    "name": location["name"],
                    "city_code": location["iataCode"],
                    "lat": float(location["geoCode"]["latitude"]),
                    "lon": float(location["geoCode"]["longitude"])
                }
                city_options.append(city_info)
                print(f"{i+1}. {location['name']} ({location['iataCode']})")
            
            return city_options
        else:
            print(f"'{city_name}' için şehir bilgisi bulunamadı.")
            return None
    except Exception as e:
        print(f"Şehir arama hatası: {str(e)}")
        return None


# ---------------------- BUDGET CALCULATION ---------------------- #
def estimate_flight_cost(from_city, to_city):
    """Uçuş maliyeti tahmini yapar"""
    # Gerçek API olmadığı için simüle ediyoruz
    base_cost = random.randint(80, 300)
    distance_factor = hash(from_city + to_city) % 20  # Simüle mesafe faktörü
    return base_cost + distance_factor * 5

def estimate_food_cost(city_code):
    """Şehirdeki ortalama günlük yemek maliyetini tahmin eder"""
    # Şehir koduna göre basit bir tahmin
    city_cost_factors = {
        'IST': 30, 'LON': 60, 'PAR': 55, 'ROM': 45, 'NYC': 70, 
        'BCN': 40, 'BER': 45, 'AMS': 50, 'VIE': 48, 'PRG': 35
    }
    return city_cost_factors.get(city_code, 40)  # Bilinmeyen şehirler için 40€ varsayalım

def estimate_activity_cost(city_code):
    """Şehirdeki günlük aktivite masrafları tahmini"""
    # Şehir koduna göre basit bir tahmin
    city_activity_factors = {
        'IST': 20, 'LON': 50, 'PAR': 45, 'ROM': 35, 'NYC': 60, 
        'BCN': 30, 'BER': 35, 'AMS': 40, 'VIE': 38, 'PRG': 25
    }
    return city_activity_factors.get(city_code, 30)  # Bilinmeyen şehirler için 30€ varsayalım

def get_hotel_average_price(city_code, access_token):
    """Şehirdeki ortalama otel fiyatını alır"""
    url = "https://test.api.amadeus.com/v1/shopping/hotel-offers"
    headers = {
        "Authorization": f"Bearer {access_token}"
    }
    params = {
        "cityCode": city_code,
        "adults": 1,
        "roomQuantity": 1,
        "bestRateOnly": True
    }

    try:
        response = requests.get(url, headers=headers, params=params)
        data = response.json()
        
        if "data" in data and len(data.get("data", [])) > 0:
            prices = []
            for offer in data.get("data", []):
                try:
                    # Access price correctly according to Amadeus structure
                    # Price is in offers[0].price.total
                    price = float(offer["offers"][0]["price"]["total"])
                    prices.append(price)
                except (KeyError, ValueError):
                    pass
            
            if prices:
                return sum(prices) / len(prices)
        
        # API yanıt vermezse veya veri yoksa tahmin değeri
        return estimate_hotel_cost(city_code)
    except Exception:
        return estimate_hotel_cost(city_code)

def estimate_hotel_cost(city_code):
    """Otel maliyeti tahmini yapar"""
    # Şehir koduna göre basit bir tahmin
    city_hotel_factors = {
        'IST': 80, 'LON': 150, 'PAR': 140, 'ROM': 110, 'NYC': 200, 
        'BCN': 100, 'BER': 110, 'AMS': 120, 'VIE': 115, 'PRG': 90
    }
    return city_hotel_factors.get(city_code, 100)  # Bilinmeyen şehirler için 100€ varsayalım

def calculate_trip_cost(itinerary, start_city, days_per_city, access_token):
    """Toplam seyahat maliyetini hesaplar"""
    total_cost = 0
    current_city = start_city
    trip_details = []
    
    # Her şehir için maliyet hesapla
    for city_info in itinerary:
        city_code = city_info["city_code"]
        city_name = city_info["name"]
        days = days_per_city[city_code]
        
        # Uçuş maliyeti
        flight_cost = estimate_flight_cost(current_city, city_code)
        
        # Konaklama maliyeti
        hotel_cost = get_hotel_average_price(city_code, access_token) * days
        
        # Yemek maliyeti
        food_cost = estimate_food_cost(city_code) * days
        
        # Aktivite maliyeti
        activity_cost = estimate_activity_cost(city_code) * days
        
        # Şehir için toplam maliyet
        city_total = flight_cost + hotel_cost + food_cost + activity_cost
        
        trip_details.append({
            "city": city_name,
            "code": city_code,
            "days": days,
            "flight_cost": flight_cost,
            "hotel_cost": hotel_cost,
            "food_cost": food_cost,
            "activity_cost": activity_cost,
            "city_total": city_total
        })
        
        total_cost += city_total
        current_city = city_code
    
    return total_cost, trip_details

def optimize_trip_duration(cities, min_days, max_days, budget, start_city, access_token):
    """Bütçeye uygun en iyi tatil süresini optimize eder"""
    best_plans = []
    
    # Her şehir için minimum 1 gün ayıralım
    base_days = {city["city_code"]: 1 for city in cities}
    total_base_days = len(cities)
    
    # Kalan günleri dağıtmak için kombinasyonları deneyelim
    remaining_days = min(max_days, budget // 100) - total_base_days  # Günlük min 100€ varsayımı
    
    if remaining_days <= 0:
        # Minimum gün sayısı ile plan oluştur
        cost, details = calculate_trip_cost(cities, start_city, base_days, access_token)
        if cost <= budget:
            best_plans.append({"total_cost": cost, "details": details, "days": sum(base_days.values())})
    else:
        # Farklı gün dağılımları deneyelim
        for _ in range(10):  # En iyi 10 farklı kombinasyonu deneyelim
            days_per_city = base_days.copy()
            
            # Rastgele gün dağılımı yapalım
            extra_days = remaining_days
            while extra_days > 0:
                city = random.choice(cities)["city_code"]
                days_per_city[city] += 1
                extra_days -= 1
            
            # Toplam maliyeti hesaplayalım
            cost, details = calculate_trip_cost(cities, start_city, days_per_city, access_token)
            
            # Bütçeye uygun planları kaydedelim
            if cost <= budget:
                best_plans.append({
                    "total_cost": cost, 
                    "details": details, 
                    "days": sum(days_per_city.values())
                })
    
    # En uygun planları bütçe kullanımı maksimum olacak şekilde sıralayalım
    best_plans.sort(key=lambda x: (x["days"], -x["total_cost"]))
    return best_plans

# ---------------------- A* / DIJKSTRA ALGORITHM ---------------------- #
def calculate_distance(city1, city2):
    """Calculate the approximate distance between two cities using coordinates"""
    # Haversine formula for distance calculation
    R = 6371  # Earth's radius in km
    lat1, lon1 = math.radians(city1["lat"]), math.radians(city1["lon"])
    lat2, lon2 = math.radians(city2["lat"]), math.radians(city2["lon"])
    
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    distance = R * c
    
    return distance

def find_optimal_route(start_city, cities):
    """Find the optimal route between cities using Dijkstra's algorithm"""
    # Create a graph representation of cities
    all_cities = [start_city] + cities
    n = len(all_cities)
    
    # Create distance matrix
    distances = []
    for i in range(n):
        row = []
        for j in range(n):
            if i == j:
                row.append(0)
            else:
                # Calculate distance between cities
                dist = calculate_distance(all_cities[i], all_cities[j])
                # Convert to travel cost (simplified: €1 per 10km)
                cost = dist * 0.1
                row.append(cost)
        distances.append(row)
    
    # Implement TSP using a greedy approach
    current = 0  # Start from the first city
    route = [current]
    unvisited = set(range(1, n))
    total_cost = 0
    
    while unvisited:
        next_city = min(unvisited, key=lambda x: distances[current][x])
        total_cost += distances[current][next_city]
        current = next_city
        route.append(current)
        unvisited.remove(current)
    
    # Return to start city to complete the circuit if needed
    # route.append(0)
    # total_cost += distances[current][0]
    
    # Map indices back to city objects
    optimal_route = [all_cities[i] for i in route]
    
    return optimal_route, total_cost

# ---------------------- CONSTRAINT SATISFACTION PROBLEM ---------------------- #
def csp_travel_planner(cities, budget, min_days, max_days, access_token):
    """
    Implement CSP for travel planning
    
    Parameters:
    - cities: List of city objects
    - budget: Total budget in euros
    - min_days: Minimum trip duration
    - max_days: Maximum trip duration
    """
    valid_plans = []
    
    # Define constraints
    def check_constraints(days_allocation, total_cost):
        # Constraint 1: Total days must be within the allowed range
        total_days = sum(days_allocation.values())
        if total_days < min_days or total_days > max_days:
            return False
        
        # Constraint 2: Total cost must be within budget
        if total_cost > budget:
            return False
            
        # Constraint 3: Each city must have at least 1 day
        for days in days_allocation.values():
            if days < 1:
                return False
                
        return True
    
    # Generate day allocations
    def backtrack(city_index, days_allocation, remaining_days, current_cities):
        # Base case: all cities have been allocated days
        if city_index == len(cities):
            # Calculate the cost of this allocation
            cost, details = calculate_trip_cost(current_cities, "START", days_allocation, access_token)
            
            # Check if this allocation meets all constraints
            if check_constraints(days_allocation, cost):
                valid_plans.append({
                    "days_allocation": days_allocation.copy(),
                    "total_cost": cost,
                    "details": details,
                    "days": sum(days_allocation.values())
                })
            return
            
        city = cities[city_index]
        city_code = city["city_code"]
        
        # Try allocating different number of days to the current city
        for days in range(1, min(remaining_days + 1, 5)):  # Max 4 days per city
            days_allocation[city_code] = days
            current_cities.append(city)
            backtrack(city_index + 1, days_allocation, remaining_days - days, current_cities)
            current_cities.pop()
            
    # Start the backtracking search
    backtrack(0, {city["city_code"]: 0 for city in cities}, max_days, [])
    
    # Sort valid plans by maximizing days and minimizing cost difference from budget
    valid_plans.sort(key=lambda x: (x["days"], -abs(budget - x["total_cost"])), reverse=True)
    
    return valid_plans

# ---------------------- GENETIC ALGORITHM ---------------------- #
def genetic_algorithm_travel_planner(cities, budget, min_days, max_days, access_token, population_size=30, generations=15):
    """
    Optimize travel plans using a genetic algorithm
    
    Parameters:
    - cities: List of city objects
    - budget: Total budget in euros
    - min_days: Minimum trip duration
    - max_days: Maximum trip duration
    """
    def create_individual():
        """Create a random travel plan"""
        # Randomly shuffle cities order
        shuffled_cities = cities.copy()
        random.shuffle(shuffled_cities)
        
        # More strategic day allocation to improve valid solution probability
        days_allocation = {}
        remaining_days = random.randint(min_days, max_days)
        
        # First ensure minimum 1 day per city
        for city in shuffled_cities:
            city_code = city["city_code"]
            days_allocation[city_code] = 1
            remaining_days -= 1
        
        # Then distribute remaining days more evenly
        while remaining_days > 0:
            city_code = random.choice([city["city_code"] for city in shuffled_cities])
            if days_allocation[city_code] < 4:  # Max 4 days per city
                days_allocation[city_code] += 1
                remaining_days -= 1
        
        return {
            "cities": shuffled_cities,
            "days": days_allocation
        }
    
    def calculate_fitness(individual):
        """Calculate the fitness of a travel plan"""
        cost, details = calculate_trip_cost(
            individual["cities"], 
            "START", 
            individual["days"], 
            access_token
        )
        
        # Fitness criteria:
        # 1. Must be within budget (plans over budget get low fitness)
        # 2. Should maximize days
        # 3. Should use as much of the budget as possible
        
        total_days = sum(individual["days"].values())
        
        if cost > budget:
            # Penalize for exceeding budget
            return 1 / (1 + (cost - budget))
        
        # For plans within budget, reward for:
        # - More days (weighted heavily)
        # - Using more of the available budget
        fitness = (total_days / max_days) * 0.7 + (cost / budget) * 0.3
        
        return fitness
    
    def crossover(parent1, parent2):
        """Create a new travel plan by combining aspects of two parents"""
        # Child inherits city order from parent1
        child_cities = parent1["cities"].copy()
        
        # Days allocation is a mix of both parents
        child_days = {}
        for city in child_cities:
            city_code = city["city_code"]
            # 50% chance to inherit from each parent
            if random.random() < 0.5:
                child_days[city_code] = parent1["days"][city_code]
            else:
                child_days[city_code] = parent2["days"][city_code]
        
        return {
            "cities": child_cities,
            "days": child_days
        }
    
    def mutate(individual):
        """Randomly modify a travel plan"""
        # 30% chance to shuffle city order
        if random.random() < 0.3:
            random.shuffle(individual["cities"])
        
        # 50% chance to modify days allocation
        if random.random() < 0.5:
            city_code = random.choice(list(individual["days"].keys()))
            
            # Add or subtract a day
            if random.random() < 0.5 and individual["days"][city_code] > 1:
                individual["days"][city_code] -= 1
            elif sum(individual["days"].values()) < max_days:
                individual["days"][city_code] += 1
        
        return individual
    
    # Create initial population
    population = [create_individual() for _ in range(population_size)]
    
    # Evolution over generations
    best_solution = None
    best_fitness = 0
    
    for generation in range(generations):
        # Calculate fitness for each individual
        fitness_scores = []
        for ind in population:
            cost, details = calculate_trip_cost(
                ind["cities"], 
                "START", 
                ind["days"], 
                access_token
            )
            
            # Store the cost with the individual for later use
            ind["cost"] = cost
            ind["details"] = details
            
            # Calculate fitness
            total_days = sum(ind["days"].values())
            
            if cost <= budget and min_days <= total_days <= max_days:
                # Valid solution gets higher fitness
                fitness = 0.7 * (total_days / max_days) + 0.3 * (cost / budget)
                
                # Keep track of best solution
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_solution = copy.deepcopy(ind)
            else:
                # Invalid solutions get lower fitness but still possible to be selected
                if cost > budget:
                    # Penalize for exceeding budget
                    fitness = 0.1 / (1 + (cost - budget) / 100)
                else:
                    # Penalize for day constraints
                    fitness = 0.1 / (1 + abs(total_days - min_days))
            
            fitness_scores.append(fitness)
        
        # Tournament selection
        new_population = []
        for _ in range(population_size):
            tournament_size = 3
            tournament = random.sample(range(population_size), tournament_size)
            winner = max(tournament, key=lambda i: fitness_scores[i])
            new_population.append(copy.deepcopy(population[winner]))
        
        # Apply crossover and mutation
        for i in range(0, population_size, 2):
            if i+1 < population_size and random.random() < 0.7:  # 70% crossover rate
                child1 = crossover(new_population[i], new_population[i+1])
                child2 = crossover(new_population[i+1], new_population[i])
                new_population[i] = child1
                new_population[i+1] = child2
                
        for i in range(population_size):
            if random.random() < 0.3:  # 30% mutation rate
                new_population[i] = mutate(new_population[i])
        
        population = new_population
    
    # If no valid solution found after all generations, create a basic valid solution
    if best_solution is None:
        print("Genetik algoritma uygun plan bulamadı, temel bir plan oluşturuluyor...")
        days_per_city = max_days // len(cities)
        remaining_days = max_days - days_per_city * len(cities)
        
        basic_solution = {
            "cities": cities.copy(),
            "days": {city["city_code"]: days_per_city for city in cities}
        }
        
        # Distribute remaining days
        for i in range(remaining_days):
            basic_solution["days"][cities[i % len(cities)]["city_code"]] += 1
        
        cost, details = calculate_trip_cost(
            basic_solution["cities"], 
            "START", 
            basic_solution["days"], 
            access_token
        )
        
        if cost <= budget:
            best_solution = basic_solution
            best_solution["cost"] = cost
            best_solution["details"] = details
    
    # Return the best solutions
    final_plans = []
    if best_solution:
        final_plans.append({
            "total_cost": best_solution["cost"], 
            "details": best_solution["details"], 
            "days": sum(best_solution["days"].values()),
            "cities_order": best_solution["cities"]
        })
        
    # Also add some diverse backup plans if available
    diverse_plans = []
    for ind in population:
        if "cost" in ind and ind["cost"] <= budget and min_days <= sum(ind["days"].values()) <= max_days:
            plan = {
                "total_cost": ind["cost"], 
                "details": ind["details"],
                "days": sum(ind["days"].values()),
                "cities_order": ind["cities"]
            }
            diverse_plans.append(plan)
    
    # Add diverse plans to final plans
    diverse_plans.sort(key=lambda x: (x["days"], -abs(budget - x["total_cost"])), reverse=True)
    final_plans.extend(diverse_plans[:4])  # Add up to 4 more diverse plans
        
    return final_plans[:5]  # Return top 5 plans

# ---------------------- ANA FONKSİYON ---------------------- #
def tatil_planla(city, lat, lon, city_code):
    print(f"=== {city} için Tatil Planı ===\n")

    get_foursquare_places(city)
    get_weather(lat, lon)

    # Otel araması için yeni bir token alalım
    client_id = "YhrTcYlqUCh5vG7AnRQATxxNaZMAZZPH"
    client_secret = "XWAbTdoc4a2aGyeQ"
    token = get_amadeus_access_token(client_id, client_secret)
    print(f"Oteller {city_code} kodu ile aranıyor...")
    get_hotels(city_code, token)


def main():
    print("=== Tatil Planlayıcı ===")
    print("İstediğiniz şehirler için tatil planı yapabilirsiniz.")
    
    # Amadeus token'ını başlangıçta alıyoruz
    client_id = "YhrTcYlqUCh5vG7AnRQATxxNaZMAZZPH"
    client_secret = "XWAbTdoc4a2aGyeQ"
    access_token = get_amadeus_access_token(client_id, client_secret)
    
    # Initialize city recommendations dictionary
    city_recommendations = {}
    
    while True:
        print("\n1. Gelişmiş Tatil Planlama (Optimizasyonlu)")
        print("q. Çıkış")
        secim = input("Seçiminiz: ")
        
        if secim == 'q':
            print("Programdan çıkılıyor...")
            break
        
        if secim == '1':
            print("\n=== Gelişmiş Tatil Planlama ===")
            print("Optimum rotayı, bütçe ve süre planlamasını birlikte yapacağız.")
            
            # Bulunduğunuz şehir
            start_city_name = input("\nBulunduğunuz şehir: ")
            start_city_options = search_city(start_city_name, access_token)
            if not start_city_options:
                print("Başlangıç şehri bulunamadı.")
                continue
                
            start_city_idx = int(input("Lütfen bir şehir seçin (numara): ")) - 1
            start_city = start_city_options[start_city_idx]
            
            # Bütçe bilgisini alalım
            try:
                budget = float(input("Toplam bütçeniz (€): "))
                if budget <= 0:
                    print("Bütçe pozitif bir değer olmalıdır.")
                    continue
            except ValueError:
                print("Lütfen geçerli bir bütçe değeri girin.")
                continue
            
            # Tatil gün aralığını alalım
            try:
                min_days = int(input("Minimum tatil süresi (gün): "))
                max_days = int(input("Maksimum tatil süresi (gün): "))
                if min_days <= 0 or max_days <= 0 or min_days > max_days:
                    print("Geçersiz gün aralığı. Minimum gün sayısı pozitif ve maksimum günden küçük olmalıdır.")
                    continue
            except ValueError:
                print("Lütfen geçerli bir gün sayısı girin.")
                continue
                
            # Gidilecek şehirler
            destinations = []
            city_count = int(input("\nKaç şehir gezmek istiyorsunuz? (2-5): "))
            if city_count < 2 or city_count > 5:
                print("2 ile 5 arasında bir değer girin.")
                continue
                
            for i in range(city_count):
                dest_name = input(f"\n{i+1}. şehir: ")
                dest_options = search_city(dest_name, access_token)
                
                if not dest_options:
                    print(f"Şehir bulunamadı: {dest_name}")
                    continue
                
                dest_idx = int(input("Lütfen bir şehir seçin (numara): ")) - 1
                destinations.append(dest_options[dest_idx])
                
                # Get recommendations for each city
                city_code = dest_options[dest_idx]["city_code"]
                city_name = dest_options[dest_idx]["name"]
                print(f"Şehir için öneriler alınıyor: {city_name}...")
                city_recommendations[city_code] = get_foursquare_places_return(city_name)
            
            print("\nKapsamlı tatil planınız hesaplanıyor...")
            print("Bu işlem biraz zaman alabilir...")
            
            # 1. En kısa yol algoritması (Dijkstra) - optimum rota belirlemek için
            print("\n1/4: Optimum rota hesaplanıyor...")
            optimal_route, route_cost = find_optimal_route(start_city, destinations)
            
            print("\n=== Optimum Seyahat Rotası ===")
            print(f"Başlangıç: {start_city['name']} ({start_city['city_code']})")
            
            for i, city in enumerate(optimal_route[1:]):
                print(f"{i+1}. {city['name']} ({city['city_code']})")
                
            print(f"Tahmini rota maliyeti: {route_cost:.2f}€")
            
            # 2. Çoklu şehir optimizasyonu - gün dağılımı için
            print("\n2/4: Bütçeye uygun süre dağılımı hesaplanıyor...")
            optimized_plans = optimize_trip_duration(
                destinations, 
                min_days, 
                max_days, 
                budget, 
                start_city['city_code'],
                access_token
            )
            
            # 3. Kısıt tabanlı seyahat planlama (CSP) - kısıtları sağlayan alternatif planlar için
            print("\n3/4: Kısıt tabanlı alternatif planlar hesaplanıyor...")
            csp_plans = csp_travel_planner(
                destinations, 
                budget, 
                min_days, 
                max_days, 
                access_token
            )
            
            # 4. Genetik algoritma - en optimum planı bulmak için
            print("\n4/4: Genetik algoritma ile plan optimizasyonu yapılıyor...")
            ga_plans = genetic_algorithm_travel_planner(
                destinations, 
                budget, 
                min_days, 
                max_days, 
                access_token
            )
            
            # Tüm sonuçları birleştirip karşılaştırma yapalım
            print("\n=== KAPSAMLI TATİL PLAN ANALİZİ ===")
            
            # Dijkstra ile belirlenen optimum rota
            print("\n🗺️ OPTIMUM ROTA:")
            print(f"Başlangıç: {start_city['name']} ({start_city['city_code']})")
            for i, city in enumerate(optimal_route[1:]):
                print(f"{i+1}. {city['name']} ({city['city_code']})")
            
            # En iyi planlar
            print("\n💰 BÜTÇE DOSTU PLAN:")
            if optimized_plans:
                best_budget_plan = min(optimized_plans, key=lambda x: x['total_cost'])
                print(f"Toplam: {best_budget_plan['total_cost']:.2f}€, Süre: {best_budget_plan['days']} gün")
                for city in best_budget_plan['details']:
                    print(f"  • {city['city']} ({city['code']}): {city['days']} gün, {city['city_total']:.2f}€")
            else:
                print("Bütçeye uygun plan bulunamadı")
                
            print("\n⏱️ MAKSIMUM DENEYIM PLANI:")
            if ga_plans:
                best_exp_plan = max(ga_plans, key=lambda x: x['days'])
                print(f"Toplam: {best_exp_plan['total_cost']:.2f}€, Süre: {best_exp_plan['days']} gün")
                cities_order = [city["name"] for city in best_exp_plan["cities_order"]]
                print(f"  Önerilen Rota: {' -> '.join(cities_order)}")
                for city in best_exp_plan['details']:
                    print(f"  • {city['city']} ({city['code']}): {city['days']} gün")
                    print(f"    Konaklama: {city['hotel_cost']:.2f}€, Uçuş: {city['flight_cost']:.2f}€")
            else:
                print("Genetik algoritmada uygun plan bulunamadı")
                
            print("\n🏆 EN DEĞERLI ZAMAN PLANI:")
            all_plans = []
            if optimized_plans:
                all_plans.extend(optimized_plans)
            if csp_plans:
                all_plans.extend(csp_plans)
            if ga_plans:
                all_plans.extend(ga_plans)
                
            if all_plans:
                # Değer skoru: günlük maliyet düşük + toplam gün yüksek
                best_value_plan = max(all_plans, key=lambda x: (x['days'] / max_days) - (x['total_cost'] / budget) / 2)
                print(f"Toplam: {best_value_plan['total_cost']:.2f}€, Süre: {best_value_plan['days']} gün")
                print(f"Günlük ortalama: {best_value_plan['total_cost'] / best_value_plan['days']:.2f}€")
                for city in best_value_plan['details']:
                    print(f"  • {city['city']} ({city['code']}): {city['days']} gün")
                    print(f"    Konaklama: {city['hotel_cost']:.2f}€ ({city['hotel_cost']/city['days']:.2f}€/gün)")
                    print(f"    Yemek: {city['food_cost']:.2f}€, Aktivite: {city['activity_cost']:.2f}€")
            else:
                print("Kriterlere uygun plan bulunamadı")
            
            # Detaylı rapor sunalım
            print("\n📊 DETAYLI RAPOR GÖRMEK İSTER MİSİNİZ? (e/h)")
            detay = input("> ")
            if detay.lower() == 'e':
                print("\n=== DETAYLI SEYAHAT RAPORU ===")
                
                if ga_plans:
                    best_plan = ga_plans[0] # Genetik algoritmanın en iyi sonucu
                    
                    print("\n📆 GÜN GÜN PLAN:")
                    current_day = 1
                    
                    for city in best_plan['details']:
                        city_code = city['code']
                        print(f"\n🏙️ {city['city']} ({city_code}) - {city['days']} gün")
                        
                        # Show Foursquare recommendations for this city
                        if city_code in city_recommendations and city_recommendations[city_code]:
                            print("  📍 Önerilen Yerler:")
                            for place in city_recommendations[city_code][:3]:  # Show top 3 places
                                print(f"    • {place}")
                        else:
                            print("  📍 Önerilen yerler yüklenemedi.")
                        
                        for day in range(city['days']):
                            print(f"\n  Gün {current_day}:")
                            if day == 0:
                                print(f"    • Sabah: Varış ve otel girişi")
                                print(f"    • Öğleden sonra: Şehir turu")
                                print(f"    • Akşam: Yerel restoranda akşam yemeği")
                            else:
                                print(f"    • Sabah: Şehir keşfi veya müze ziyareti")
                                print(f"    • Öğleden sonra: Yerel aktiviteler")
                                print(f"    • Akşam: Gezi ve yemek deneyimi")
                            current_day += 1
                    
                    print(f"\n💸 TOPLAM BÜTÇE DAĞILIMI:")
                    total = best_plan['total_cost']
                    flight_total = sum(city['flight_cost'] for city in best_plan['details'])
                    hotel_total = sum(city['hotel_cost'] for city in best_plan['details'])
                    food_total = sum(city['food_cost'] for city in best_plan['details'])
                    activity_total = sum(city['activity_cost'] for city in best_plan['details'])
                    
                    print(f"  • Uçuşlar: {flight_total:.2f}€ ({flight_total/total*100:.1f}%)")
                    print(f"  • Konaklama: {hotel_total:.2f}€ ({hotel_total/total*100:.1f}%)")
                    print(f"  • Yemek: {food_total:.2f}€ ({food_total/total*100:.1f}%)")
                    print(f"  • Aktiviteler: {activity_total:.2f}€ ({activity_total/total*100:.1f}%)")
                    print(f"  • TOPLAM: {total:.2f}€ / {budget:.2f}€ (Bütçenin {total/budget*100:.1f}%'i)")
                else:
                    print("Detaylı rapor için yeterli veri bulunamadı.")
        else:
            print("Geçersiz seçim.")

# Programı çalıştır
if __name__ == "__main__":
    main()
