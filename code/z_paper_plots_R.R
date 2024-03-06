
# Dependencies -----------------------------------------------------------------

### libraries
library(tidyverse)
library(lubridate)
library(ggplot2)

### helper functions 
PlotFormatter <- function() {
  theme_minimal(base_family = "Times New Roman") +
    theme(
      axis.ticks = element_line(size = 0.5, color = "black"),
      plot.title = element_text(hjust = 0.5),
      plot.subtitle = element_text(hjust = 0.5),
      plot.background = element_rect(fill = "white"),
      panel.background = element_blank(),
      axis.line = element_line(color = "black"),
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank()
    )
}

### data 

sales = read_csv('sales_clean.csv')

# Section 2 --------------------------------------------------------------------

### trend

plot_data = sales %>%
  group_by(date = floor_date(date, "month"), family) %>%
  summarise(sales = sum(sales, na.rm = TRUE)) %>%
  filter(family == "BEAUTY")

ggplot(plot_data) +
  geom_line(aes(x = date, y = sales), linewidth = .25) +
  scale_x_date(date_breaks = "1 year") + 
  labs(
    x = "",
    y = "Sales",
    title = "Monthly Sales of Beauty Products",
    tag = "Figure 1"
  ) +
  PlotFormatter()

ggsave(dpi = 300, height = 5, width = 7, "pics/trend_example.png")

### seasonality

plot_data = sales %>%
  filter(store_nbr == 1 & family == "PRODUCE") %>%
  filter(date >= "2016-03-01" & date <= "2016-03-31")

ggplot(plot_data) +
  geom_line(aes(x = date, y = sales), linewidth = .25) +
  scale_x_date(date_breaks = "1 week") + 
  labs(
    x = "",
    y = "Sales",
    title = "Weekly Seasonality in Produce Sales",
    tag = "Figure 2"
  ) +
  PlotFormatter()

ggsave(dpi = 300, height = 5, width = 7, "pics/seas_example.png")

### percent change

plot_data = sales %>%
  group_by(date = floor_date(date, "month"), family) %>%
  summarise(sales = sum(sales, na.rm = TRUE)) %>%
  ungroup() %>%
  filter(family == "BEAUTY") %>%
  mutate(
    sales_prcnt_chg = (sales - dplyr::lag(sales)) / dplyr::lag(sales)
  )

ggplot(plot_data) +
  geom_line(aes(x = date, y = sales_prcnt_chg), linewidth = .25) +
  scale_x_date(date_breaks = "1 year") + 
  labs(
    x = "",
    y = "Percent Change in Sales",
    title = "Percent Change in Sales of Beauty Products",
    tag = "Figure 3"
  ) +
  PlotFormatter()

ggsave(dpi = 300, height = 5, width = 7, "pics/prcnt_chge_example.png")

# Section 3 --------------------------------------------------------------------

plot_data = sales %>%
  group_by(date, family) %>%
  summarise(sales = sum(sales, na.rm = TRUE)) %>%
  filter(family %in% c('PERSONAL CARE', 'PET SUPPLIES', 'POULTRY', 'MEATS', 'MAGAZINES', 'PRODUCE', 'BOOKS', 'CLEANING')) %>%
  filter(date >= "2016-01-01")

ggplot(plot_data) +
  geom_line(aes(x = date, y = sales), linewidth = .25) +
  scale_x_date(date_breaks = "5 month") + 
  labs(
    x = "",
    y = "Sales",
    tag = "Figure 4"
  ) +
  facet_wrap(~family, scales = 'free', ncol = 2) +
  PlotFormatter()

ggsave(dpi = 300, height = 10, width = 10, "pics/multi_series.png")

# Section 4 --------------------------------------------------------------------

### acf 1

acf_data = acf(
  plot_data %>% filter(family == "PRODUCE") %>% pull(sales),
  28,
  plot = FALSE
)

plot_df = data.frame(lag = acf_data$lag[-1], acf = acf_data$acf[-1])

ggplot(plot_df) +
  geom_bar(
    aes(x = lag, y = acf),
    stat = "identity",
    position = "dodge"
  ) +
  geom_hline(yintercept = 0) +
  labs(
    x = "Lag",
    title = "Produce Sales Autocorrelations, One Month of Sales",
    y = "Autocorrelation",
    tag = "Figure 5"
  ) +
  PlotFormatter()

ggsave(dpi = 300, height = 5, width = 7, "pics/acf.png")

### acf 2

acf_data = acf(
  plot_data %>% filter(family == "PRODUCE") %>% pull(sales),
  365,
  plot = FALSE
)

plot_df = data.frame(lag = acf_data$lag[-1], acf = acf_data$acf[-1])

ggplot(plot_df) +
  geom_bar(
    aes(x = lag, y = acf),
    stat = "identity",
    position = "dodge"
  ) +
  geom_hline(yintercept = 0) +
  labs(
    x = "Lag",
    title = "Produce Sales Autocorrelations, One Year of Sales",
    y = "Autocorrelation",
    tag = "Figure 6"
  ) +
  PlotFormatter()

ggsave(dpi = 300, height = 5, width = 7, "pics/acf2.png")