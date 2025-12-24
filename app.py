import streamlit as st
import pandas as pd
import joblib
import numpy as np
from pathlib import Path
import plotly.graph_objects as go

# Configuration de la page
st.set_page_config(
    page_title="⚽ FIFA-Style Player Value Predictor",
    page_icon="⚽",
    layout="wide"
)

# CSS personnalisé pour ressembler à FIFA
st.markdown("""
<style>
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #0055FF 0%, #00AAFF 100%);
    }
    .metric-container {
        background-color: #0E1117;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #262730;
    }
    .fifa-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 15px;
        padding: 20px;
        border: 2px solid #00AAFF;
        box-shadow: 0 4px 15px rgba(0, 170, 255, 0.2);
    }
    .stat-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 10px;
        padding: 8px;
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
    }
    .stat-name {
        font-weight: bold;
        color: #FFFFFF;
        font-size: 14px;
    }
    .stat-value {
        font-weight: bold;
        color: #00AAFF;
        font-size: 18px;
    }
</style>
""", unsafe_allow_html=True)

# Titre FIFA-style
st.markdown("""
<div style="text-align: center; padding: 20px;">
    <h1 style="background: linear-gradient(90deg, #0055FF, #00AAFF); 
               -webkit-background-clip: text; 
               -webkit-text-fill-color: transparent;
               font-size: 3em;
               margin-bottom: 10px;">
        ⚽ FIFA PLAYER VALUE
    </h1>
    <h3 style="color: #CCCCCC; margin-top: 0;">
        Market Value Predictor - FIFA Ratings Style
    </h3>
</div>
""", unsafe_allow_html=True)

# Sidebar FIFA-style
with st.sidebar:
    st.markdown('<div class="fifa-card">', unsafe_allow_html=True)
    st.markdown("### 🎮 FIFA STYLE")
    st.markdown("---")
    
    # Sélection de la position
    POSITIONS = {
        "ATTACKER ⚽": "attacker",
        "MIDFIELDER 🎯": "midfielder", 
        "DEFENDER 🛡️": "defender",
        "GOALKEEPER 🧤": "goalkeeper"
    }
    
    selected_position_display = st.selectbox(
        "SELECT PLAYER POSITION:",
        list(POSITIONS.keys()),
        key="position_select"
    )
    selected_position = POSITIONS[selected_position_display]
    
    st.markdown("---")
    
    # Info modèle
    st.markdown("**MODEL INFO:**")
    
    @st.cache_resource
    def load_model_data(position):
        try:
            model_path = Path(f"models/best_model_{position}.pkl")
            if model_path.exists():
                return joblib.load(model_path)
        except Exception as e:
            st.error(f"Error: {str(e)}")
            return None
    
    model_data = load_model_data(selected_position)
    if model_data:
        st.success(f"✅ {selected_position.upper()} Model Loaded")
        st.info(f"📊 Features: {len(model_data['features'])}")
        st.info(f"🤖 Type: {type(model_data['model']).__name__}")
    
    st.markdown("---")
    st.caption("Based on FIFA ratings data")
    st.markdown('</div>', unsafe_allow_html=True)

# Fonction pour créer un slider FIFA-style
def fifa_slider(label, min_val=0, max_val=99, default=50, key=None, help_text=""):
    """Crée un slider avec style FIFA"""
    return st.slider(
        label,
        min_value=min_val,
        max_value=max_val,
        value=default,
        step=1,
        key=key,
        help=help_text,
        format="%d ⭐"
    )

# Section principale
if model_data:
    features_list = model_data['features']
    
    # Organiser UNIQUEMENT les features de l'attaquant
    if selected_position == "attacker":
        # Liste complète des features de l'attaquant
        attacker_features = [
            'overall_rating', 'potential', 'wage_euro', 'international_reputation(1-5)',
            'skill_moves(1-5)', 'national_rating', 'finishing', 'short_passing',
            'volleys', 'dribbling', 'curve', 'ball_control', 'reactions',
            'shot_power', 'long_shots', 'positioning', 'vision', 'composure'
        ]
        
        # Vérifier que toutes les features sont dans la liste
        missing_features = set(features_list) - set(attacker_features)
        if missing_features:
            st.warning(f"Features manquantes dans la liste: {missing_features}")
            # Ajouter les features manquantes
            for feat in missing_features:
                if feat not in attacker_features:
                    attacker_features.append(feat)
        
        # Grouper les features sans doublons
        categories = {
            "📊 BASIC RATINGS": ['overall_rating', 'potential', 'national_rating'],
            "⚽ ATTACKING": ['finishing', 'volleys', 'shot_power', 'long_shots'],
            "🎯 SKILL": ['dribbling', 'curve', 'ball_control', 'skill_moves(1-5)'],
            "🧠 MENTAL": ['positioning', 'vision', 'composure', 'reactions'],
            "🔄 TECHNIQUE": ['short_passing']
        }
    elif selected_position == "midfielder":
        # Features pour milieu
        categories = {
            "📊 BASIC RATINGS": ['overall_rating', 'potential'],
            "🎯 PASSING": ['short_passing', 'long_passing', 'vision'],
            "⚽ TECHNIQUE": ['dribbling', 'ball_control', 'long_shots'],
            "🧠 MENTAL": ['positioning', 'composure', 'reactions']
        }
    elif selected_position == "defender":
        # Features pour défenseur
        categories = {
            "📊 BASIC RATINGS": ['overall_rating', 'potential'],
            "🛡️ DEFENDING": ['heading_accuracy', 'interceptions', 'marking', 
                            'standing_tackle', 'sliding_tackle'],
            "🧠 MENTAL": ['composure', 'reactions'],
            "⚽ TECHNIQUE": ['short_passing', 'ball_control']
        }
    else:  # goalkeeper
        # Features pour gardien
        categories = {
            "📊 BASIC RATINGS": ['overall_rating', 'potential'],
            "🧤 GOALKEEPING": ['reactions']
        }
    
    # Interface en onglets
    tab1, tab2, tab3 = st.tabs(["📊 PLAYER RATINGS", "💰 CONTRACT", "🎯 PREDICTION"])
    
    with tab1:
        st.markdown(f'<div class="fifa-card">', unsafe_allow_html=True)
        st.markdown(f"### {selected_position_display.replace('⚽', '').strip()} RATINGS")
        st.markdown("---")
        
        input_values = {}
        
        # Initialiser les valeurs par défaut
        default_values = {
            'overall_rating': 82,
            'potential': 86,
            'wage_euro': 50000,
            'international_reputation(1-5)': 3,
            'skill_moves(1-5)': 4,
            'national_rating': 81,
            'finishing': 85,
            'short_passing': 80,
            'volleys': 78,
            'dribbling': 84,
            'curve': 79,
            'ball_control': 83,
            'reactions': 82,
            'shot_power': 83,
            'long_shots': 80,
            'positioning': 83,
            'vision': 79,
            'composure': 82,
            'long_passing': 75,
            'heading_accuracy': 70,
            'interceptions': 75,
            'marking': 78,
            'standing_tackle': 80,
            'sliding_tackle': 75
        }
        
        # Afficher par catégories
        for category, features in categories.items():
            # Filtrer les features qui sont dans notre liste de features
            available_features = [f for f in features if f in features_list]
            
            if available_features:
                st.subheader(f"{category}")
                
                # Créer des colonnes (maximum 3)
                cols = st.columns(min(3, len(available_features)))
                
                for idx, feature in enumerate(available_features):
                    with cols[idx % len(cols)]:
                        # Déterminer le type d'input
                        if feature in ['international_reputation(1-5)', 'skill_moves(1-5)']:
                            # Slider 1-5 pour réputation/skill
                            min_val, max_val = (1, 5)
                            default = default_values.get(feature, 3)
                            
                            # Créer un ID unique avec la position
                            unique_key = f"slider_{feature}_{selected_position}_{idx}"
                            
                            input_values[feature] = st.slider(
                                feature.replace('_', ' ').replace('(1-5)', '').upper(),
                                min_value=min_val,
                                max_value=max_val,
                                value=default,
                                step=1,
                                key=unique_key,
                                help=f"{feature} rating"
                            )
                            
                        elif feature == 'wage_euro':
                            # Input spécial pour salaire (sera dans l'onglet Contract)
                            continue
                            
                        else:
                            # Slider normal 0-99
                            default = default_values.get(feature, 75)
                            if 'overall' in feature or 'potential' in feature or 'national' in feature:
                                default = default_values.get(feature, 80)
                            
                            # Créer un ID unique avec la position
                            unique_key = f"slider_{feature}_{selected_position}_{idx}"
                            
                            input_values[feature] = fifa_slider(
                                feature.replace('_', ' ').upper(),
                                min_val=0,
                                max_val=99,
                                default=default,
                                key=unique_key
                            )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="fifa-card">', unsafe_allow_html=True)
        st.markdown("### 💰 CONTRACT & REPUTATION")
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Salaire
            if 'wage_euro' in features_list:
                input_values['wage_euro'] = st.number_input(
                    "WEEKLY WAGE (€)",
                    min_value=1000,
                    max_value=1000000,
                    value=50000,
                    step=1000,
                    key=f"wage_{selected_position}"
                )
                st.metric("Annual Salary", f"€ {input_values.get('wage_euro', 0) * 52:,.0f}")
            
            # Réputation internationale
            if 'international_reputation(1-5)' in features_list:
                st.markdown("### INTERNATIONAL REPUTATION")
                
                # Si déjà défini dans l'onglet précédent, l'utiliser
                current_value = input_values.get('international_reputation(1-5)', 3)
                
                rep_value = st.select_slider(
                    "Stars",
                    options=[1, 2, 3, 4, 5],
                    value=current_value,
                    key=f"reputation_{selected_position}"
                )
                input_values['international_reputation(1-5)'] = rep_value
                
                # Afficher les étoiles
                stars = "⭐" * rep_value
                st.markdown(f"<h2 style='text-align: center; color: gold;'>{stars}</h2>", unsafe_allow_html=True)
        
        with col2:
            # Skill Moves (pour attaquant seulement)
            if selected_position == "attacker" and 'skill_moves(1-5)' in features_list:
                st.markdown("### SKILL MOVES")
                
                # Si déjà défini dans l'onglet précédent, l'utiliser
                current_value = input_values.get('skill_moves(1-5)', 4)
                
                skill_value = st.select_slider(
                    "Skill Stars",
                    options=[1, 2, 3, 4, 5],
                    value=current_value,
                    key=f"skill_{selected_position}"
                )
                input_values['skill_moves(1-5)'] = skill_value
                
                # Afficher les étoiles de skill
                skill_stars = "🌟" * skill_value
                st.markdown(f"<h2 style='text-align: center; color: #00FFAA;'>{skill_stars}</h2>", unsafe_allow_html=True)
            
            # Note nationale (si présente)
            if 'national_rating' in features_list:
                st.markdown("### NATIONAL TEAM RATING")
                
                # Si déjà défini dans l'onglet précédent, l'utiliser
                current_value = input_values.get('national_rating', 81)
                
                input_values['national_rating'] = fifa_slider(
                    "RATING",
                    min_val=0,
                    max_val=99,
                    default=current_value,
                    key=f"national_{selected_position}"
                )
        
        # Afficher un résumé
        st.markdown("---")
        st.markdown("### 📝 CONTRACT SUMMARY")
        
        summary_col1, summary_col2, summary_col3 = st.columns(3)
        
        with summary_col1:
            if 'overall_rating' in input_values:
                st.metric("Overall", f"{input_values['overall_rating']}")
        
        with summary_col2:
            if 'potential' in input_values:
                st.metric("Potential", f"{input_values['potential']}")
        
        with summary_col3:
            if 'wage_euro' in input_values:
                st.metric("Weekly Wage", f"€{input_values['wage_euro']:,}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        st.markdown('<div class="fifa-card">', unsafe_allow_html=True)
        st.markdown("### 🎯 VALUE PREDICTION")
        st.markdown("---")
        
        # Bouton de prédiction
        if st.button("🚀 CALCULATE MARKET VALUE", 
                    type="primary", 
                    use_container_width=True,
                    key=f"predict_{selected_position}"):
            
            try:
                # S'assurer que toutes les features ont une valeur
                for feature in features_list:
                    if feature not in input_values:
                        # Valeur par défaut selon le type de feature
                        if feature == 'wage_euro':
                            input_values[feature] = 50000
                        elif feature in ['international_reputation(1-5)', 'skill_moves(1-5)']:
                            input_values[feature] = 3
                        elif 'rating' in feature:
                            input_values[feature] = 80
                        else:
                            input_values[feature] = 75
                
                # Créer DataFrame
                input_df = pd.DataFrame([input_values])
                input_df = input_df[features_list]  # Bon ordre
                
                # Faire la prédiction
                with st.spinner("🔄 Analyzing FIFA ratings..."):
                    prediction = model_data['model'].predict(input_df)[0]
                    prediction_millions = prediction / 1000000
                
                # Afficher le résultat
                st.markdown("---")
                
                # Carte de joueur FIFA-style
                st.markdown("""
                <div style='background: linear-gradient(135deg, #0F2027, #203A43, #2C5364);
                            border-radius: 20px;
                            padding: 30px;
                            text-align: center;
                            border: 3px solid #00AAFF;
                            box-shadow: 0 10px 30px rgba(0, 170, 255, 0.3);'>
                """, unsafe_allow_html=True)
                
                # Position et rating
                overall = input_values.get('overall_rating', 80)
                st.markdown(f"""
                <div style='display: flex; justify-content: center; align-items: center; gap: 40px;'>
                    <div style='background-color: #0055FF; 
                                border-radius: 50%; 
                                width: 100px; 
                                height: 100px;
                                display: flex;
                                align-items: center;
                                justify-content: center;
                                border: 4px solid white;'>
                        <h1 style='color: white; margin: 0;'>{overall}</h1>
                    </div>
                    <div style='text-align: left;'>
                        <h2 style='color: white; margin: 0;'>{selected_position_display}</h2>
                        <h3 style='color: #00AAFF; margin: 5px 0;'>FIFA RATED PLAYER</h3>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Valeur prédite
                st.markdown(f"""
                <div style='margin-top: 30px;'>
                    <h4 style='color: #CCCCCC; margin-bottom: 10px;'>PREDICTED MARKET VALUE</h4>
                    <h1 style='color: #00FFAA; 
                               font-size: 3.5em; 
                               margin: 0;
                               text-shadow: 0 0 10px rgba(0, 255, 170, 0.5);'>
                        €{prediction_millions:,.1f}M
                    </h1>
                    <p style='color: #999999;'>(€{prediction:,.0f})</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Métriques supplémentaires
                st.markdown("---")
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                
                with metric_col1:
                    if 'potential' in input_values and 'overall_rating' in input_values:
                        growth = input_values['potential'] - input_values['overall_rating']
                        st.metric("📈 POTENTIAL GROWTH", f"+{growth} Points")
                
                with metric_col2:
                    if 'wage_euro' in input_values:
                        annual_wage = input_values['wage_euro'] * 52
                        ratio = prediction / annual_wage if annual_wage > 0 else 0
                        st.metric("💰 VALUE/WAGE RATIO", f"{ratio:.1f}x")
                
                with metric_col3:
                    # Catégorie de valeur
                    if prediction_millions < 10:
                        category = "YOUNG TALENT"
                        color = "#00AAFF"
                    elif prediction_millions < 30:
                        category = "FIRST TEAM PLAYER"
                        color = "#00FFAA"
                    elif prediction_millions < 60:
                        category = "ELITE PLAYER"
                        color = "#FFAA00"
                    elif prediction_millions < 100:
                        category = "SUPERSTAR"
                        color = "#FF5500"
                    else:
                        category = "WORLD CLASS"
                        color = "#FF0055"
                    
                    st.markdown(f"""
                    <div style='background-color: {color}20; 
                                padding: 10px; 
                                border-radius: 10px;
                                border-left: 5px solid {color};'>
                        <p style='color: {color}; 
                                  font-weight: bold;
                                  margin: 0;
                                  font-size: 16px;'>
                            {category}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Tableau des stats
                with st.expander("📋 COMPLETE STATS BREAKDOWN"):
                    # Créer un DataFrame avec toutes les stats
                    all_stats = []
                    for feature in features_list:
                        if feature in input_values:
                            all_stats.append({
                                'STAT': feature.replace('_', ' ').upper(),
                                'RATING': input_values[feature],
                                'TYPE': 'Rating' if feature not in ['wage_euro'] else 'Contract'
                            })
                    
                    if all_stats:
                        stats_df = pd.DataFrame(all_stats)
                        st.dataframe(stats_df, use_container_width=True, hide_index=True)
                        
                        # Bouton d'export
                        csv = stats_df.to_csv(index=False)
                        st.download_button(
                            label="📥 DOWNLOAD PLAYER CARD",
                            data=csv,
                            file_name=f"fifa_player_{selected_position}.csv",
                            mime="text/csv",
                            key=f"download_{selected_position}"
                        )
                
            except Exception as e:
                st.error(f"❌ Prediction error: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
        
        # Exemples pour chaque position
        st.markdown("---")
        with st.expander("🎮 FIFA RATING EXAMPLES"):
            examples = {
                "attacker": {
                    "Player Type": "World Class Striker",
                    "Overall": "90-94",
                    "Key Stats": "Finishing 90+, Dribbling 88+, Composure 89+",
                    "Expected Value": "€100M+"
                },
                "midfielder": {
                    "Player Type": "Creative Playmaker", 
                    "Overall": "88-92",
                    "Key Stats": "Vision 90+, Passing 88+, Dribbling 86+",
                    "Expected Value": "€80M+"
                },
                "defender": {
                    "Player Type": "Elite Center Back",
                    "Overall": "86-90",
                    "Key Stats": "Defending 88+, Physical 85+, Composure 86+",
                    "Expected Value": "€60M+"
                },
                "goalkeeper": {
                    "Player Type": "Top Goalkeeper",
                    "Overall": "87-91", 
                    "Key Stats": "Reactions 89+, Positioning 88+",
                    "Expected Value": "€50M+"
                }
            }
            
            ex = examples.get(selected_position, {})
            if ex:
                for key, value in ex.items():
                    st.write(f"**{key}:** {value}")
        
        st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666666; padding: 20px;">
    <p>FIFA-STYLE PLAYER VALUE PREDICTOR • BASED ON MACHINE LEARNING MODELS</p>
    <p style="font-size: 0.8em;">This is a demonstration tool. Values are predictions based on FIFA ratings.</p>
</div>
""", unsafe_allow_html=True)

# Bouton pour réinitialiser
if st.button("🔄 NEW PREDICTION", type="secondary", key="reset_button"):
    st.rerun()