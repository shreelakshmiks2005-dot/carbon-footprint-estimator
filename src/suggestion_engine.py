"""
Suggestion Engine - Provides personalized sustainability recommendations
"""

class SuggestionEngine:
    """Generate personalized carbon reduction suggestions"""
    
    def __init__(self):
        self.suggestions_db = self._build_suggestions_database()
    
    def _build_suggestions_database(self):
        """Build database of suggestions based on different categories"""
        return {
            'transportation': {
                'high': [
                    "Switch to public transportation for your daily commute - can save up to 50% emissions",
                    "Consider carpooling with colleagues to reduce daily travel emissions",
                    "Upgrade to an electric or hybrid vehicle for your next car purchase",
                    "Try biking or walking for short distances (< 3 km)",
                    "Reduce flights by using video conferencing for meetings"
                ],
                'medium': [
                    "Optimize your driving route to reduce fuel consumption",
                    "Maintain your vehicle regularly for better fuel efficiency",
                    "Consider one flight-free month annually"
                ],
                'low': [
                    "Great job! Your transportation emissions are below average",
                    "Encourage others to adopt sustainable transportation habits"
                ]

            },
            'diet': {
                'high': [
                    "Introduce Meatless Mondays - save ~6 kg CO2 per week",
                    "Reduce meat consumption - consider plant-based alternatives",
                    "Choose chicken or fish over beef and lamb (lower emissions)",
                    "Eat more local, seasonal foods to reduce transportation emissions",
                    "Try a flexitarian diet - reduce meat consumption gradually"
                ],
                'medium': [
                    "Replace some meals with vegetarian options",
                    "Buy local produce when possible",
                    "Choose sustainably sourced seafood"
                ],
                'low': [
                    "Excellent! Your diet has a low carbon footprint",
                    "Share your sustainable eating tips with friends and family"
                ]

            },
            'energy': {
                'high': [
                    "Switch to renewable energy providers - can save up to 80% emissions",
                    "Install LED bulbs throughout your home (75% less energy than incandescent)",
                    "Use a programmable thermostat to optimize heating/cooling",
                    "Unplug devices when not in use - eliminate phantom power drain",
                    "Consider installing solar panels for long-term savings",
                    "Upgrade to Energy Star appliances"
                ],
                'medium': [
                    "Reduce thermostat by 1-2°C in winter (3-5% savings)",
                    "Use cold water for laundry when possible",
                    "Air dry clothes instead of using a dryer"
                ],
                'low': [
                    "Your energy consumption is efficient - well done!",
                    "Continue monitoring your usage through smart meters"
                ]
            },
            'waste': {
                'high': [
                    "Increase recycling rate to at least 80% of waste",
                    "Compost organic waste - diverts from landfills",
                    "Buy secondhand items instead of new whenever possible",
                    "Reduce single-use plastics - switch to reusable bags and bottles",
                    "Buy in bulk to reduce packaging waste"
                ],
                'medium': [
                    "Join local recycling programs if available",
                    "Donate used clothes instead of discarding them",
                    "Support companies with sustainable packaging"
                ],
                'low': [
                    "Excellent waste management practices!",
                    "Help educate your community about recycling"
                ]
            },
            'water': {
                'high': [
                    "Install low-flow showerheads - save 2,700 gallons/year",
                    "Fix leaky taps and pipes promptly",
                    "Take shorter showers (5 min or less)",
                    "Use full loads for laundry and dishwashing",
                    "Install a dual-flush toilet"
                ],
                'medium': [
                    "Reduce shower time by 1-2 minutes",
                    "Turn off tap while brushing teeth",
                    "Water plants with collected rainwater"
                ],
                'low': [
                    "Great water conservation habits!",
                    "Share water-saving tips with neighbors"
                ]
            }
        }
    
    def generate_suggestions(self, user_data, carbon_footprint, avg_carbon_footprint):
        """
        Generate personalized suggestions based on user data
        
        Args:
            user_data: Dictionary with user's lifestyle data
            carbon_footprint: User's predicted carbon footprint
            avg_carbon_footprint: Average carbon footprint for comparison
        
        Returns:
            Dictionary with categorized suggestions
        """
        suggestions = {
            'overall': self._get_overall_suggestions(carbon_footprint, avg_carbon_footprint),
            'transportation': self._get_transportation_suggestions(user_data),
            'diet': self._get_diet_suggestions(user_data),
            'energy': self._get_energy_suggestions(user_data),
            'waste': self._get_waste_suggestions(user_data),
            'water': self._get_water_suggestions(user_data)
        }
        
        return suggestions
    
    def _get_overall_suggestions(self, footprint, avg_footprint):
        """Overall carbon footprint suggestions"""
        if footprint > avg_footprint * 1.2:
            return {
                'status': '⚠️ Above Average',
                'message': f'Your carbon footprint ({footprint:.1f} kg CO2/month) is {((footprint/avg_footprint - 1) * 100):.1f}% above average. See specific recommendations below.',
                'priority': 'high'
            }
        elif footprint < avg_footprint * 0.8:
            return {
                'status': '✓ Below Average',
                'message': f'Excellent! Your carbon footprint ({footprint:.1f} kg CO2/month) is {((1 - footprint/avg_footprint) * 100):.1f}% below average.',
                'priority': 'low'
            }
        else:
            return {
                'status': '~ Average',
                'message': f'Your carbon footprint ({footprint:.1f} kg CO2/month) is close to average. Small changes can make a big difference!',
                'priority': 'medium'
            }
    
    def _get_transportation_suggestions(self, user_data):
        """Transportation-specific suggestions"""
        level = self._assess_transportation_level(user_data)
        suggestions = self.suggestions_db['transportation'][level]
        
        return {
            'level': level,
            'current_habits': self._describe_transportation(user_data),
            'suggestions': suggestions,
            'potential_savings': self._estimate_transport_savings(user_data)
        }
    
    def _get_diet_suggestions(self, user_data):
        """Diet-specific suggestions"""
        level = self._assess_diet_level(user_data)
        suggestions = self.suggestions_db['diet'][level]
        
        return {
            'level': level,
            'current_habits': self._describe_diet(user_data),
            'suggestions': suggestions,
            'potential_savings': self._estimate_diet_savings(user_data)
        }
    
    def _get_energy_suggestions(self, user_data):
        """Energy-specific suggestions"""
        level = self._assess_energy_level(user_data)
        suggestions = self.suggestions_db['energy'][level]
        
        return {
            'level': level,
            'current_habits': self._describe_energy(user_data),
            'suggestions': suggestions,
            'potential_savings': self._estimate_energy_savings(user_data)
        }
    
    def _get_waste_suggestions(self, user_data):
        """Waste-specific suggestions"""
        level = self._assess_waste_level(user_data)
        suggestions = self.suggestions_db['waste'][level]
        
        return {
            'level': level,
            'current_habits': self._describe_waste(user_data),
            'suggestions': suggestions,
            'potential_savings': self._estimate_waste_savings(user_data)
        }
    
    def _get_water_suggestions(self, user_data):
        """Water-specific suggestions"""
        level = self._assess_water_level(user_data)
        suggestions = self.suggestions_db['water'][level]
        
        return {
            'level': level,
            'current_habits': self._describe_water(user_data),
            'suggestions': suggestions,
            'potential_savings': self._estimate_water_savings(user_data)
        }
    
    # Assessment methods
    def _assess_transportation_level(self, user_data):
        score = 0
        if user_data.get('transportation_mode') == 'Car':
            score += 3
        elif user_data.get('transportation_mode') == 'Public Transport':
            score += 1
        
        if user_data.get('commute_distance_km', 0) > 20:
            score += 2
        
        if user_data.get('flights_per_year', 0) > 2:
            score += 2
        
        return 'high' if score >= 5 else ('medium' if score >= 2 else 'low')
    
    def _assess_diet_level(self, user_data):
        if user_data.get('diet_type') == 'Vegan':
            return 'low'
        elif user_data.get('diet_type') == 'Vegetarian':
            return 'medium'
        else:
            return 'high' if user_data.get('meat_consumption_kg_per_month', 0) > 10 else 'medium'
    
    def _assess_energy_level(self, user_data):
        consumption = user_data.get('electricity_usage_kwh', 0) + user_data.get('gas_usage_therms', 0) * 2.93
        avg_consumption = 400 + 50 * 2.93  # Average ~550 kWh equivalent
        
        return 'high' if consumption > avg_consumption * 1.2 else ('low' if consumption < avg_consumption * 0.8 else 'medium')
    
    def _assess_waste_level(self, user_data):
        recycling = user_data.get('recycling_percentage', 0)
        clothes = user_data.get('new_clothes_per_month', 0)
        
        score = recycling / 10 - clothes
        
        return 'low' if score > 5 else ('high' if score < 2 else 'medium')
    
    def _assess_water_level(self, user_data):
        usage = user_data.get('water_usage_gallons', 0)
        avg_usage = 2500  # Average monthly household usage
        
        return 'high' if usage > avg_usage * 1.2 else ('low' if usage < avg_usage * 0.8 else 'medium')
    
    # Description methods
    def _describe_transportation(self, user_data):
        desc = []
        if user_data.get('transportation_mode'):
            desc.append(f"Primary transport: {user_data['transportation_mode']}")
        if user_data.get('commute_distance_km'):
            desc.append(f"Daily commute: {user_data['commute_distance_km']:.1f} km")
        if user_data.get('flights_per_year'):
            desc.append(f"Flights per year: {user_data['flights_per_year']}")
        return " | ".join(desc) if desc else "No data available"
    
    def _describe_diet(self, user_data):
        desc = []
        if user_data.get('diet_type'):
            desc.append(f"Diet: {user_data['diet_type']}")
        if user_data.get('meat_consumption_kg_per_month'):
            desc.append(f"Meat consumption: {user_data['meat_consumption_kg_per_month']:.1f} kg/month")
        return " | ".join(desc) if desc else "No data available"
    
    def _describe_energy(self, user_data):
        desc = []
        if user_data.get('electricity_usage_kwh'):
            desc.append(f"Electricity: {user_data['electricity_usage_kwh']:.1f} kWh")
        if user_data.get('gas_usage_therms'):
            desc.append(f"Gas: {user_data['gas_usage_therms']:.1f} therms")
        return " | ".join(desc) if desc else "No data available"
    
    def _describe_waste(self, user_data):
        desc = []
        if user_data.get('recycling_percentage') is not None:
            desc.append(f"Recycling: {user_data['recycling_percentage']:.0f}%")
        if user_data.get('new_clothes_per_month'):
            desc.append(f"New clothes: {user_data['new_clothes_per_month']:.1f}/month")
        return " | ".join(desc) if desc else "No data available"
    
    def _describe_water(self, user_data):
        if user_data.get('water_usage_gallons'):
            return f"Water usage: {user_data['water_usage_gallons']:.0f} gallons/month"
        return "No data available"
    
    # Savings estimation methods
    def _estimate_transport_savings(self, user_data):
        """Estimate potential CO2 savings from transportation changes"""
        if user_data.get('transportation_mode') == 'Car':
            return f"~{user_data.get('commute_distance_km', 10) * 20 * 0.16:.1f} kg CO2/month by switching to public transport"
        return "~5-10 kg CO2/month possible"
    
    def _estimate_diet_savings(self, user_data):
        """Estimate potential CO2 savings from diet changes"""
        if user_data.get('diet_type') == 'Meat-based':
            return "~20-30 kg CO2/month possible with flexitarian diet"
        return "~5-10 kg CO2/month possible"
    
    def _estimate_energy_savings(self, user_data):
        """Estimate potential CO2 savings from energy efficiency"""
        current = user_data.get('electricity_usage_kwh', 500) * 0.92
        return f"~{current * 0.3:.1f} kg CO2/month with renewable energy + efficiency"
    
    def _estimate_waste_savings(self, user_data):
        """Estimate potential CO2 savings from waste reduction"""
        clothes = user_data.get('new_clothes_per_month', 2)
        return f"~{clothes * 4:.1f} kg CO2/month by reducing new clothes purchases"
    
    def _estimate_water_savings(self, user_data):
        """Estimate potential CO2 savings from water conservation"""
        return "~2-5 kg CO2/month with water-saving measures"


if __name__ == '__main__':
    # Test suggestion engine
    engine = SuggestionEngine()
    
    # Sample user data
    user_data = {
        'transportation_mode': 'Car',
        'commute_distance_km': 25,
        'flights_per_year': 4,
        'diet_type': 'Meat-based',
        'meat_consumption_kg_per_month': 12,
        'electricity_usage_kwh': 750,
        'gas_usage_therms': 80,
        'water_usage_gallons': 4000,
        'recycling_percentage': 50,
        'new_clothes_per_month': 3,
        'car_type': 'Petrol'
    }
    
    # Generate suggestions
    suggestions = engine.generate_suggestions(user_data, 200, 180)
    
    print("="*60)
    print("PERSONALIZED CARBON REDUCTION SUGGESTIONS")
    print("="*60)
    
    for category, content in suggestions.items():
        print(f"\n{category.upper()}")
        print("-" * 40)
        if isinstance(content, dict):
            for key, value in content.items():
                if isinstance(value, list):
                    for item in value:
                        print(f"  • {item}")
                else:
                    print(f"  {key.replace('_', ' ').title()}: {value}")
        else:
            print(f"  {content}")
