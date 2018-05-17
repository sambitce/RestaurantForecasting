import React, {Component}  from 'react' ;
import {Tabs, Tab,section,Grid,Cell,Card,CardTitle, CardText,CardActions,Button,CardMenu,IconButton,CardMedia} from 'react-mdl';

class Forecast extends Component{

state = {

/*
  gender: undefined,
  username: undefined ,
  error:undefined
  */
  no_of_visitors:undefined,
  model:undefined
}
  callapi = async(e) => {

    e.preventDefault();
    const forecast_date = e.target.elements.forecast_date.value;
    const model_no = e.target.elements.model.value;
    console.log(forecast_date);
   console.log(model_no);
    const api_call = await fetch (`http://localhost:5000/forecast?forecast_date=${forecast_date}&model=${model_no}`);
    const data = await api_call.json();
    console.log(data);
    this.setState({
      /*
        gender: data.results[0].gender,
        username: data.results[0].id.name  */
        no_of_visitors: data.forecast

    })
  //console.log(this.state.gender)

  }
  constructor(props){
    super(props);
    this.state = {activeTab: 0};

  }


toggleCategories(){
  if( this.state.activeTab === 0 ){
    return(
      <div className="forecast-grid" >
      <Card shadow={5} style={{minWidth: '450' , margin: 'auto'}}>
        <CardTitle style={{color: '#fff' ,height: '176px' , background: 'url(https://kaggle2.blob.core.windows.net/competitions/kaggle/7277/logos/header.png) center / cover' }}>
          Visitors Data
        </CardTitle>
        <CardText> Predict how many future visitors a restaurant will receive </CardText>
        <CardActions border  >
          <Button colored href="https://www.kaggle.com/c/recruit-restaurant-visitor-forecasting/data">Kaggle</Button>
        </CardActions>
        <CardMenu style={{color: '#fff'}} >
           <IconButton name="share" />
        </CardMenu>
      </Card>
      <Card shadow={5} style={{minWidth: '550' , margin: 'auto' }}>

        <CardTitle style={{color: '#fff' ,height: '200px'  ,  background: 'url(/day_chart.png) center / cover'  }}>

        </CardTitle>
        <CardText> Average number of visitors on a week day</CardText>
        <CardActions border>
          <Button colored>Kaggle</Button>
        </CardActions>
        <CardMenu style={{color: '#fff'}} >
           <IconButton name="share" />
        </CardMenu>
      </Card>

      <Card shadow={5} style={{minWidth: '550' , margin: 'auto' }}>

        <CardTitle style={{color: '#fff' ,height: '200px'  ,  background: 'url(/GPS_map.png) center / cover'  }}>

        </CardTitle>
        <CardText> Average number of visitors bsed on a location</CardText>
        <CardActions border>
          <Button colored>Kaggle</Button>
        </CardActions>
        <CardMenu style={{color: '#fff'}} >
           <IconButton name="share" />
        </CardMenu>
      </Card>

      <Card shadow={5} style={{minWidth: '550' , margin: 'auto' }}>

        <CardTitle style={{color: '#fff' ,height: '200px'  ,  background: 'url(/MonthlyChart.png) center / cover'  }}>

        </CardTitle>
        <CardText> Monthly Average number of visitors </CardText>
        <CardActions border>
          <Button colored>Kaggle</Button>
        </CardActions>
        <CardMenu style={{color: '#fff'}} >
           <IconButton name="share" />
        </CardMenu>
      </Card>
      </div>

    )
  }
  else if( this.state.activeTab === 1 ){


    return(
      <form onSubmit={this.callapi}  >
        <label > Select Forecast Date: </label>
        <input type="date" name="forecast_date"  placeholder="Forecast Date"  />

        <button name="model" value="model1" > Get Forecast </button>
        <p> Expected Visitors: {this.state.no_of_visitors} </p>
      </form>
    )
  }
  else if( this.state.activeTab === 2 ){
    return(

        <form onSubmit={this.callapi}  >
          <label > Select Forecast Date: </label>
          <input type="date" name="forecast_date"  placeholder="Forecast Date"  />

          <button name="model" value="model2" > Get Forecast </button>
          <p> Expected Visitors: {this.state.no_of_visitors} </p>
        </form>
    )
  }
  else if( this.state.activeTab === 3 ){
    return(

        <form onSubmit={this.callapi}  >
          <label > Select Forecast Date: </label>
          <input type="date" name="forecast_date"  placeholder="Forecast Date"  />

          <button name="model" value="model3" > Get Forecast </button>
          <p> Expected Visitors: {this.state.no_of_visitors} </p>
        </form>
    )
  }
}

  render(){

    return (
      <div className="category-tabs">
        <Tabs activeTab={this.state.activeTab} onChange={(tabId) => this.setState({activeTab: tabId})} ripple   >
          <Tab>Data</Tab>
          <Tab>Model1</Tab>
          <Tab>Model2</Tab>
          <Tab>Model3</Tab>
        </Tabs>

        <section className="forecast-grid" >
        <Grid  >
          <Cell col={12}>
            <div className="content">
          {this.toggleCategories()}
          </div>
          </Cell>
          </Grid>
        </section>
      </div>
    )
  }
}

export default Forecast;
